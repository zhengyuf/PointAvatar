import os
import torch
import numpy as np
import cv2
import json
import imageio
import skimage
from tqdm import tqdm


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_folder,
                 subject_name,
                 json_name,
                 sub_dir,
                 img_res,
                 is_eval,
                 subsample=1,
                 hard_mask=False,
                 only_json=False,
                 use_mean_expression=False,
                 use_var_expression=False,
                 use_background=False,
                 load_images=False,
                 ):
        """
        sub_dir: list of scripts/testing subdirectories for the subject, e.g. [MVI_1810, MVI_1811]
        Data structure:
            RGB images in data_folder/subject_name/subject_name/sub_dir[i]/image
            foreground masks in data_folder/subject_name/subject_name/sub_dir[i]/mask
            json files containing FLAME parameters in data_folder/subject_name/subject_name/sub_dir[i]/json_name
        json file structure:
            frames: list of dictionaries, which are structured like:
                file_path: relative path to image
                world_mat: camera extrinsic matrix (world to camera). Camera rotation is actually the same for all frames,
                           since the camera is fixed during capture.
                           The FLAME head is centered at the origin, scaled by 4 times.
                expression: 50 dimension expression parameters
                pose: 15 dimension pose parameters
                flame_keypoints: 2D facial keypoints calculated from FLAME
            shape_params: 100 dimension FLAME shape parameters, shared by all scripts and testing frames of the subject
            intrinsics: camera focal length fx, fy and the offsets of the principal point cx, cy
        img_res: a list containing height and width, e.g. [256, 256] or [512, 512]
        subsample: subsampling the images to reduce frame rate, mainly used for inference and evaluation
        hard_mask: whether to use boolean segmentation mask or not
        only_json: used for testing, when there is no GT images or masks. If True, only load json.
        use_background: if False, replace with white background. Otherwise, use original background
        load_images: if True, load images at the beginning instead of at each iteration
        use_mean_expression: if True, use mean expression of the training set as the canonical expression
        use_var_expression: if True, blendshape regularization weight will depend on the variance of expression
                            (more regularization if variance is small in the training set.)
        """
        sub_dir = [str(dir) for dir in sub_dir]
        self.img_res = img_res
        self.use_background = use_background
        self.load_images = load_images
        self.hard_mask = hard_mask

        self.data = {
            "image_paths": [],
            "mask_paths": [],
            # camera extrinsics
            "world_mats": [],
            # FLAME expression and pose parameters
            "expressions": [],
            "flame_pose": [],
            # saving image names and subdirectories
            "img_name": [],
            "sub_dir": [],
        }

        for dir in sub_dir:
            instance_dir = os.path.join(data_folder, subject_name, subject_name, dir)
            assert os.path.exists(instance_dir), "Data directory {} is empty".format(instance_dir)

            cam_file = '{0}/{1}'.format(instance_dir, json_name)

            with open(cam_file, 'r') as f:
                camera_dict = json.load(f)
            for frame in camera_dict['frames']:
                # world to camera matrix
                world_mat = np.array(frame['world_mat']).astype(np.float32)
                # camera to world matrix
                self.data["world_mats"].append(world_mat)
                self.data["expressions"].append(np.array(frame['expression']).astype(np.float32))
                self.data["flame_pose"].append(np.array(frame['pose']).astype(np.float32))
                self.data["sub_dir"].append(dir)
                image_path = '{0}/{1}.png'.format(instance_dir, frame["file_path"])
                self.data["image_paths"].append(image_path)
                self.data["mask_paths"].append(image_path.replace('image', 'mask'))
                self.data["img_name"].append(int(frame["file_path"].split('/')[-1]))


        self.gt_dir = instance_dir
        self.shape_params = torch.tensor(camera_dict['shape_params']).float().unsqueeze(0)
        focal_cxcy = camera_dict['intrinsics']

        if isinstance(subsample, int) and subsample > 1:
            for k, v in self.data.items():
                self.data[k] = v[::subsample]
        elif isinstance(subsample, list):
            if len(subsample) == 2:
                subsample = list(range(subsample[0], subsample[1]))
            for k, v in self.data.items():
                self.data[k] = [v[s] for s in subsample]

        self.data["expressions"] = torch.from_numpy(np.stack(self.data["expressions"], 0))
        self.data["flame_pose"] = torch.from_numpy(np.stack(self.data["flame_pose"], 0))
        self.data["world_mats"] = torch.from_numpy(np.stack(self.data["world_mats"], 0)).float()


        # construct intrinsic matrix
        intrinsics = np.zeros((4, 4))

        # from whatever camera convention to pytorch3d
        intrinsics[0, 0] = focal_cxcy[0] * 2
        intrinsics[1, 1] = focal_cxcy[1] * 2
        intrinsics[0, 2] = (focal_cxcy[2] * 2 - 1.0) * -1
        intrinsics[1, 2] = (focal_cxcy[3] * 2 - 1.0) * -1

        intrinsics[3, 2] = 1.
        intrinsics[2, 3] = 1.
        self.intrinsics = intrinsics

        if intrinsics[0, 0] < 0:
            intrinsics[:, 0] *= -1
            self.data["world_mats"][:, 0, :] *= -1
        self.data["world_mats"][:, :3, 2] *= -1
        self.data["world_mats"][:, 2, 3] *= -1

        if use_mean_expression:
            self.mean_expression = torch.mean(self.data["expressions"], 0, keepdim=True)
        else:
            self.mean_expression = torch.zeros_like(self.data["expressions"][[0], :])
        if use_var_expression:
            self.var_expression = torch.var(self.data["expressions"], 0, keepdim=True)
        else:
            self.var_expression = None

        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.only_json = only_json

        images = []
        masks = []
        if load_images and not only_json:
            print("Loading all images, this might take a while.")
            for idx in tqdm(range(len(self.data["image_paths"]))):
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1,0)).float()
                object_mask = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                if not self.use_background:
                    if not hard_mask:
                        rgb = rgb * object_mask.unsqueeze(1).float() + (1 - object_mask.unsqueeze(1).float())
                    else:
                        rgb = rgb * (object_mask.unsqueeze(1) > 0.5) + ~(object_mask.unsqueeze(1) > 0.5)
                images.append(rgb)
                masks.append(object_mask)

        self.data['images'] = images
        self.data['masks'] = masks

    def __len__(self):
        return len(self.data["image_paths"])

    def __getitem__(self, idx):
        sample = {
            "idx": torch.LongTensor([idx]),
            "img_name": torch.LongTensor([self.data["img_name"][idx]]),
            "sub_dir": self.data["sub_dir"][idx],
            "intrinsics": self.intrinsics,
            "expression": self.data["expressions"][idx],
            "flame_pose": self.data["flame_pose"][idx],
            "cam_pose": self.data["world_mats"][idx],
            }

        ground_truth = {}

        if not self.only_json:
            if not self.load_images:
                ground_truth["object_mask"] = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()
                if not self.use_background:
                    if not self.hard_mask:
                        ground_truth['rgb'] = rgb * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())
                    else:
                        ground_truth['rgb'] = rgb * (ground_truth["object_mask"].unsqueeze(1) > 0.5) + ~(ground_truth["object_mask"].unsqueeze(1) > 0.5)
                else:
                    ground_truth['rgb'] = rgb
            else:
                ground_truth = {
                    'rgb': self.data['images'][idx],
                    'object_mask': self.data['masks'][idx]
                }

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # this function is borrowed from https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
        # get list of dictionaries and returns sample, ground_truth as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    try:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    except:
                        ret[k] = [obj[k] for obj in entry]
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

def load_rgb(path, img_res):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    img = cv2.resize(img, (int(img_res[0]), int(img_res[1])))
    img = img.transpose(2, 0, 1)
    return img


def load_mask(path, img_res):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)

    alpha = cv2.resize(alpha, (int(img_res[0]), int(img_res[1])))
    object_mask = alpha / 255

    return object_mask