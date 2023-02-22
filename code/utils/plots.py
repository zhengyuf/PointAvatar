import numpy as np
import torch
import torchvision
import trimesh
from PIL import Image
import os
import cv2

SAVE_OBJ_LIST = [1]

def save_pcl_to_ply(filename, points, colors=None, normals=None):
    save_dir=os.path.dirname(os.path.abspath(filename))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if colors is not None:
        colors = colors.cpu().detach().numpy()
    if normals is not None:
        normals = normals.cpu().detach().numpy()
    mesh = trimesh.Trimesh(vertices=points.detach().cpu().numpy(),vertex_normals = normals, vertex_colors = colors)
    #there is a bug in trimesh of it only saving normals when we tell the exporter explicitly to do so for point clouds.
    #thus we are calling the exporter directly instead of mesh.export(...)
    f = open(filename, "wb")
    data = trimesh.exchange.ply.export_ply(mesh, vertex_normal=True)
    f.write(data)
    f.close()
    return


def plot(img_index, model_outputs, ground_truth, path, epoch, img_res, is_eval=False, first=False):
    # arrange data to plot
    batch_size = model_outputs['batch_size']
    plot_images(model_outputs, ground_truth, path, epoch, img_index, 1, img_res, batch_size, is_eval)

    canonical_color = torch.clamp(model_outputs['pnts_albedo'], 0., 1.)
    if not is_eval:
        return
    for idx, img_idx in enumerate(img_index):
        wo_epoch_path = path[idx].replace('/epoch_{}'.format(epoch), '')
        if img_idx in SAVE_OBJ_LIST:
            deformed_color = model_outputs["pnts_color_deformed"].reshape(batch_size, -1, 3)[idx]
            filename = '{0}/{1:04d}_deformed_color_{2}.ply'.format(wo_epoch_path, epoch, img_idx)
            save_pcl_to_ply(filename, model_outputs['deformed_points'].reshape(batch_size, -1, 3)[idx],
                            normals=model_outputs["pnts_normal_deformed"].reshape(batch_size, -1, 3)[idx],
                            colors=deformed_color)

            filename = '{0}/{1:04d}_deformed_albedo_{2}.ply'.format(wo_epoch_path, epoch, img_idx)
            save_pcl_to_ply(filename, model_outputs['deformed_points'].reshape(batch_size, -1, 3)[idx],
                            normals=model_outputs["pnts_normal_deformed"].reshape(batch_size, -1, 3)[idx],
                            colors=canonical_color)
    if first:
        wo_epoch_path = path[0].replace('/epoch_{}'.format(epoch), '')
        filename = '{0}/{1:04d}_canonical_points_albedo.ply'.format(wo_epoch_path, epoch)
        save_pcl_to_ply(filename, model_outputs["canonical_points"], colors=canonical_color)

        if 'unconstrained_canonical_points' in model_outputs:
            filename = '{0}/{1:04d}_unconstrained_canonical_points.ply'.format(wo_epoch_path, epoch)
            save_pcl_to_ply(filename, model_outputs['unconstrained_canonical_points'],
                            colors=canonical_color)
    if epoch == 0 or is_eval:
        if first:
            wo_epoch_path = path[0].replace('/epoch_{}'.format(epoch), '')
            filename = '{0}/{1:04d}_canonical_verts.ply'.format(wo_epoch_path, epoch)
            save_pcl_to_ply(filename, model_outputs['canonical_verts'].reshape(-1, 3),
                            colors=get_lbs_color(model_outputs['flame_lbs_weights']))


def plot_image(rgb, path, img_index, plot_nrow, img_res, type, fill=True):
    rgb_plot = lin2img(rgb, img_res)

    tensor = torchvision.utils.make_grid(rgb_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = np.clip(tensor, 0., 1.)
    tensor = (tensor * scale_factor).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    img = Image.fromarray(tensor)
    if not os.path.exists('{0}/{1}'.format(path, type)):
        os.mkdir('{0}/{1}'.format(path, type))
    img.save('{0}/{2}/{1}.png'.format(path, img_index, type))

    if fill:
        tensor = cv2.erode(tensor, kernel, iterations=1)
        tensor = cv2.dilate(tensor, kernel, iterations=1)

        img = Image.fromarray(tensor)
        if not os.path.exists('{0}/{1}_erode_dilate'.format(path, type)):
            os.mkdir('{0}/{1}_erode_dilate'.format(path, type))
        img.save('{0}/{2}_erode_dilate/{1}.png'.format(path, img_index, type))


def get_lbs_color(lbs_points):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')
    red = cmap.colors[5]
    cyan = cmap.colors[3]
    blue = cmap.colors[1]
    pink = [1, 1, 1]

    if lbs_points.shape[-1] == 5:
        colors = torch.from_numpy(
            np.stack([np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[
                None]).cuda()
    else:
        colors = torch.from_numpy(
            np.stack([np.array(red), np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[
                None]).cuda()
    lbs_points = (colors * lbs_points[:, :, None]).sum(1)
    return lbs_points


def plot_images(model_outputs, ground_truth, path, epoch, img_index, plot_nrow, img_res, batch_size, is_eval):
    num_samples = img_res[0] * img_res[1]
    if 'rgb' in ground_truth:
        rgb_gt = ground_truth['rgb']
        if 'rendered_landmarks' in model_outputs:
            rendered_landmarks = model_outputs['rendered_landmarks'].reshape(batch_size, num_samples, 3)
            rgb_gt = rgb_gt * (1 - rendered_landmarks) + rendered_landmarks * torch.tensor([1, 0, 0]).cuda()
    else:
        rgb_gt = None
    rgb_points = model_outputs['rgb_image']
    rgb_points = rgb_points.reshape(batch_size, num_samples, 3)

    if 'rendered_landmarks' in model_outputs:
        rendered_landmarks = model_outputs['rendered_landmarks'].reshape(batch_size, num_samples, 3)
        rgb_points_rendering = rgb_points * (1 - rendered_landmarks) + rendered_landmarks * torch.tensor([1, 0, 0]).cuda()
        output_vs_gt = rgb_points_rendering
    else:
        output_vs_gt = rgb_points

    normal_points = model_outputs['normal_image']
    normal_points = normal_points.reshape(batch_size, num_samples, 3)

    if rgb_gt is not None:
        output_vs_gt = torch.cat((output_vs_gt, rgb_gt, normal_points), dim=0)
    else:
        output_vs_gt = torch.cat((output_vs_gt, normal_points), dim=0)

    if 'shading_image' in model_outputs:
        output_vs_gt = torch.cat((output_vs_gt, model_outputs['shading_image'].reshape(batch_size, num_samples, 3)), dim=0)
        output_vs_gt = torch.cat((output_vs_gt, model_outputs['albedo_image'].reshape(batch_size, num_samples, 3)), dim=0)

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)
    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=batch_size).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)
    img = Image.fromarray(tensor)
    wo_epoch_path = path[0].replace('/epoch_{}'.format(epoch), '')
    if not os.path.exists('{0}/rendering'.format(wo_epoch_path)):
        os.mkdir('{0}/rendering'.format(wo_epoch_path))
    img.save('{0}/rendering/epoch_{1:04d}_{2}.png'.format(wo_epoch_path, epoch, img_index[0]))

    if is_eval:
        for i, idx in enumerate(img_index):
            plot_image(rgb_points[[i]], path[i], idx, plot_nrow, img_res, 'rgb')
            plot_image(normal_points[[i]], path[i], idx, plot_nrow, img_res, 'normal', fill=False)

    del output_vs_gt


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])