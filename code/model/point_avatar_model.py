import math
from functools import partial

import torch
import torch.nn as nn
from flame.FLAME import FLAME
from pytorch3d.ops import knn_points
from pytorch3d.renderer import (AlphaCompositor,
                                PerspectiveCameras,
                                PointsRasterizationSettings,
                                PointsRasterizer,
                                )
from pytorch3d.structures import Pointclouds
from model.point_cloud import PointCloud

from functorch import jacfwd, vmap

from model.geometry_network import GeometryNetwork
from model.deformer_network import ForwardDeformer
from model.texture_network import RenderingNetwork


print_flushed = partial(print, flush=True)


class PointAvatar(nn.Module):
    def __init__(self, conf, shape_params, img_res, canonical_expression, canonical_pose, use_background):
        super().__init__()
        self.FLAMEServer = FLAME('./flame/FLAME2020/generic_model.pkl', './flame/FLAME2020/landmark_embedding.npy',
                                 n_shape=100,
                                 n_exp=50,
                                 shape_params=shape_params,
                                 canonical_expression=canonical_expression,
                                 canonical_pose=canonical_pose).cuda()
        self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
            self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)
        self.prune_thresh = conf.get_float('prune_thresh', default=0.5)
        self.geometry_network = GeometryNetwork(**conf.get_config('geometry_network'))
        self.deformer_network = ForwardDeformer(FLAMEServer=self.FLAMEServer, **conf.get_config('deformer_network'))
        self.rendering_network = RenderingNetwork(**conf.get_config('rendering_network'))
        self.ghostbone = self.deformer_network.ghostbone
        if self.ghostbone:
            self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), self.FLAMEServer.canonical_transformations], 1)
        self.pc = PointCloud(**conf.get_config('point_cloud')).cuda()

        n_points = self.pc.points.shape[0]
        self.img_res = img_res
        self.use_background = use_background
        if self.use_background:
            init_background = torch.zeros(img_res[0] * img_res[1], 3).float().cuda()
            self.background = nn.Parameter(init_background)
        else:
            self.background = torch.ones(img_res[0] * img_res[1], 3).float().cuda()
        self.raster_settings = PointsRasterizationSettings(
            image_size=img_res[0],
            radius=self.pc.radius_factor * (0.75 ** math.log2(n_points / 100)),
            points_per_pixel=10
        )
        # keypoint rasterizer is only for debugging camera intrinsics
        self.raster_settings_kp = PointsRasterizationSettings(
            image_size=self.img_res[0],
            radius=0.007,
            points_per_pixel=1
        )
        self.visible_points = torch.zeros(n_points).bool().cuda()
        self.compositor = AlphaCompositor().cuda()


    def _compute_canonical_normals_and_feature_vectors(self):
        p = self.pc.points.detach()
        # randomly sample some points in the neighborhood within 0.25 distance
        eikonal_points = torch.cat([p, p + (torch.rand(p.shape).cuda() - 0.5) * 0.5], dim=0)
        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach())
        n_points = self.pc.points.shape[0]
        canonical_normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1)
        geometry_output = self.geometry_network(self.pc.points.detach())  # not using SDF to regularize point location
        sdf_values = geometry_output[:, 0]

        feature_vector = torch.sigmoid(geometry_output[:, 1:] * 10)  # albedo vector

        if self.training and hasattr(self, "_output"):
            self._output['sdf_values'] = sdf_values
            self._output['grad_thetas'] = grad_thetas
        if not self.training:
            self._output['pnts_albedo'] = feature_vector

        return canonical_normals, feature_vector

    def _render(self, point_cloud, cameras, render_kp=False):
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings if not render_kp else self.raster_settings_kp)
        fragments = rasterizer(point_cloud)
        r = rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        images, weights = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            alphas,
            point_cloud.features_packed().permute(1, 0),
        )
        images = images.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
        # batch_size, img_res, img_res, points_per_pixel
        if self.training and not render_kp:
            n_points = self.pc.points.shape[0]
            # the first point for each pixel is visible
            visible_points = fragments.idx.long()[..., 0].reshape(-1)
            visible_points = visible_points[visible_points != -1]

            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

            # points with weights larger than prune_thresh are visible
            visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > self.prune_thresh]
            visible_points = visible_points[visible_points != -1]

            n_points = self.pc.points.shape[0]
            visible_points = visible_points % n_points
            self.visible_points[visible_points] = True

        return images

    def forward(self, input):
        self._output = {}
        intrinsics = input["intrinsics"].clone()
        cam_pose = input["cam_pose"].clone()
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        batch_size = flame_pose.shape[0]
        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:
            # identity transformation for body
            transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)

        cameras = PerspectiveCameras(device='cuda', R=R, T=T, K=intrinsics)
        # make sure the cameras focal length is logged too
        focal_length = intrinsics[:, [0, 1], [0, 1]]
        cameras.focal_length = focal_length
        cameras.principal_point = cameras.get_principal_point()

        n_points = self.pc.points.shape[0]
        total_points = batch_size * n_points

        canonical_normals, feature_vector = self._compute_canonical_normals_and_feature_vectors()

        transformed_points, rgb_points, albedo_points, shading_points, normals_points = self.get_rbg_value_functorch(pnts_c=self.pc.points,
                                                                            normals=canonical_normals,
                                                                            feature_vectors=feature_vector,
                                                                            pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                            betas=expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
                                                                            transformations=transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
                                                                            )

        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach())

        transformed_points = transformed_points.reshape(batch_size, n_points, 3)
        rgb_points = rgb_points.reshape(batch_size, n_points, 3)
        # point feature to rasterize and composite
        features = torch.cat([rgb_points, torch.ones_like(rgb_points[..., [0]])], dim=-1)
        if not self.training:
            # render normal image
            normal_begin_index = features.shape[-1]
            normals_points = normals_points.reshape(batch_size, n_points, 3)
            features = torch.cat([features, normals_points * 0.5 + 0.5], dim=-1)

            shading_begin_index = features.shape[-1]
            albedo_begin_index = features.shape[-1] + 3
            albedo_points = torch.clamp(albedo_points, 0., 1.)
            features = torch.cat([features, shading_points.reshape(batch_size, n_points, 3), albedo_points.reshape(batch_size, n_points, 3)], dim=-1)

        transformed_point_cloud = Pointclouds(points=transformed_points, features=features)

        images = self._render(transformed_point_cloud, cameras)

        if not self.training:
            # render landmarks for easier camera format debugging
            landmarks2d, landmarks3d = self.FLAMEServer.find_landmarks(verts, full_pose=flame_pose)
            transformed_verts = Pointclouds(points=landmarks2d, features=torch.ones_like(landmarks2d))
            rendered_landmarks = self._render(transformed_verts, cameras, render_kp=True)

        foreground_mask = images[..., 3].reshape(-1, 1)
        if not self.use_background:
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask)
        else:
            bkgd = torch.sigmoid(self.background * 100)
            rgb_values = images[..., :3].reshape(-1, 3) + (1 - foreground_mask) * bkgd.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)

        rgb_image = rgb_values.reshape(batch_size, self.img_res[0], self.img_res[1], 3)

        # training outputs
        output = {
            'img_res': self.img_res,
            'batch_size': batch_size,
            'predicted_mask': foreground_mask,  # mask loss
            'rgb_image': rgb_image,
            'canonical_points': pnts_c_flame,
            # for flame loss
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
        }

        if not self.training:
            output_testing = {
                'normal_image': images[..., normal_begin_index:normal_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'shading_image': images[..., shading_begin_index:shading_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'albedo_image': images[..., albedo_begin_index:albedo_begin_index+3].reshape(-1, 3) + (1 - foreground_mask),
                'rendered_landmarks': rendered_landmarks.reshape(-1, 3),
                'pnts_color_deformed': rgb_points.reshape(batch_size, n_points, 3),
                'canonical_verts': self.FLAMEServer.canonical_verts.reshape(-1, 3),
                'deformed_verts': verts.reshape(-1, 3),
                'deformed_points': transformed_points.reshape(batch_size, n_points, 3),
                'pnts_normal_deformed': normals_points.reshape(batch_size, n_points, 3),
                #'pnts_normal_canonical': canonical_normals,
            }
            if self.deformer_network.deform_c:
                output_testing['unconstrained_canonical_points'] = self.pc.points
            output.update(output_testing)
        output.update(self._output)

        return output


    def get_rbg_value_functorch(self, pnts_c, normals, feature_vectors, pose_feature, betas, transformations):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)
        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])
        n_points = pnts_c.shape[0]
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature):
            pnts_c = pnts_c.unsqueeze(0)
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c)
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)
            pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights)
            pnts_d = pnts_d.reshape(-1)
            return pnts_d, pnts_d

        normals = normals.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)
        grads_batch = grads_batch.reshape(-1, batch_size, 3, 3).transpose(0, 1).reshape(-1, 3, 3)

        grads_inv = grads_batch.inverse()
        normals_d = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', normals, grads_inv), dim=1)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        # some relighting code for inference
        # rot_90 =torch.Tensor([0.7071, 0, 0.7071, 0, 1.0000, 0, -0.7071, 0, 0.7071]).cuda().float().reshape(3, 3)
        # normals_d = torch.einsum('ij, bi->bj', rot_90, normals_d)
        shading = self.rendering_network(normals_d)
        albedo = feature_vectors
        rgb_vals = torch.clamp(shading * albedo * 2, 0., 1.)
        return pnts_d, rgb_vals, albedo, shading, normals_d
