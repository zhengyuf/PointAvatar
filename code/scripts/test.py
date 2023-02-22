import os
from pyhocon import ConfigFactory
import sys
import torch

import utils.general as utils
import utils.plots as plt

from functools import partial
from model.point_avatar_model import PointAvatar

print = partial(print, flush=True)
class TestRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.conf.put('dataset.test.subsample', 1)
        self.conf.put('dataset.test.load_images', False)

        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['load_path'] != '':
            load_path = kwargs['load_path']
        else:
            load_path = self.train_dir
        assert os.path.exists(load_path)

        utils.mkdir_ifnotexists(self.eval_dir)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          is_eval=False,
                                                                                          only_json=True,
                                                                                          **self.conf.get_config('dataset.train'))

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         only_json=kwargs['only_json'],
                                                                                         use_background=self.use_background,
                                                                                         is_eval=True,
                                                                                         **self.conf.get_config('dataset.test'))

        print('Finish loading data ...')

        self.model = PointAvatar(conf=self.conf.get_config('model'),
                                shape_params=self.plot_dataset.shape_params,
                                img_res=self.plot_dataset.img_res,
                                canonical_expression=self.train_dataset.mean_expression,
                                canonical_pose=self.conf.get_float(
                                    'dataset.canonical_pose',
                                    default=0.2),
                                use_background=self.use_background)
        if torch.cuda.is_available():
            self.model.cuda()
        old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
        self.checkpoints_path = old_checkpnts_dir
        assert os.path.exists(old_checkpnts_dir)
        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
        n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
        self.model.pc.init(n_points)
        self.model.pc = self.model.pc.cuda()

        self.model.raster_settings.radius = saved_model_state['radius']

        self.model.load_state_dict(saved_model_state["model_state_dict"]) #, strict=False)
        self.start_epoch = saved_model_state['epoch']
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.optimize_inputs = self.optimize_expression or self.optimize_pose

        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=min(int(self.conf.get_int('train.max_points_training') /self.model.pc.points.shape[0]),self.conf.get_int('train.max_batch',default='10')),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )
        self.optimize_tracking = False
        if self.optimize_inputs:
            self.input_params_subdir = "TestInputParameters"
            test_input_params = []
            if self.optimize_expression:
                init_expression = self.plot_dataset.data["expressions"]

                self.expression = torch.nn.Embedding(len(self.plot_dataset), self.model.deformer_network.num_exp, _weight=init_expression, sparse=True).cuda()
                test_input_params += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(len(self.plot_dataset), 15,
                                                     _weight=self.plot_dataset.data["flame_pose"],
                                                     sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(len(self.plot_dataset), 3,
                                                      _weight=self.plot_dataset.data["world_mats"][:, :3, 3],
                                                      sparse=True).cuda()
                test_input_params += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            self.optimizer_cam = torch.optim.SparseAdam(test_input_params,
                                                        self.conf.get_float('train.learning_rate_cam'))

            try:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                if self.optimize_expression:
                    self.expression.load_state_dict(data["expression_state_dict"])
                if self.optimize_pose:
                    self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                    self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                print('Using pre-tracked test expressions')
            except:
                self.optimize_tracking = True
                from model.loss import Loss
                self.loss = Loss(mask_weight=0.0)
                print('Optimizing test expressions')

        self.img_res = self.plot_dataset.img_res


    def save_test_tracking(self, epoch):
        if not os.path.exists(os.path.join(self.checkpoints_path, "TestInputParameters")):
            os.mkdir(os.path.join(self.checkpoints_path, "TestInputParameters"))
        if self.optimize_inputs:
            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, "TestInputParameters", "latest.pth"))

    def run(self):
        self.model.eval()
        self.model.training = False
        if self.optimize_tracking:
            print("Optimizing tracking, this is a slow process which is only used for calculating metrics. \n"
                  "for qualitative animation, set optimize_expression and optimize_camera to False in the conf file.")
            for data_index, (indices, model_input, ground_truth) in enumerate(self.plot_dataloader):
                print(list(model_input["idx"].reshape(-1).cpu().numpy()))
                for k, v in model_input.items():
                    try:
                        model_input[k] = v.cuda()
                    except:
                        model_input[k] = v
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.cuda()
                    except:
                        ground_truth[k] = v

                R = model_input['cam_pose'][:, :3, :3]
                for i in range(20):
                    if self.optimize_expression:
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        model_input['cam_pose'] = torch.cat([R, self.camera_pose(model_input["idx"]).squeeze(1).unsqueeze(-1)], -1)

                    model_outputs = self.model(model_input)
                    loss_output = self.loss(model_outputs, ground_truth)
                    loss = loss_output['loss']
                    self.optimizer_cam.zero_grad()
                    loss.backward()
                    self.optimizer_cam.step()
            self.save_test_tracking(epoch=self.start_epoch)

        eval_all = True
        eval_iterator = iter(self.plot_dataloader)
        is_first_batch = True
        for img_index in range(len(self.plot_dataloader)):
            indices, model_input, ground_truth = next(eval_iterator)
            batch_size = model_input['expression'].shape[0]
            for k, v in model_input.items():
                try:
                    model_input[k] = v.cuda()
                except:
                    model_input[k] = v

            for k, v in ground_truth.items():
                try:
                    ground_truth[k] = v.cuda()
                except:
                    ground_truth[k] = v

            if self.optimize_inputs:
                if self.optimize_expression:
                    model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                if self.optimize_pose:
                    model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                    model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)

            model_outputs = self.model(model_input)
            for k, v in model_outputs.items():
                try:
                    model_outputs[k] = v.detach()
                except:
                    model_outputs[k] = v
            plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_'+str(self.start_epoch)) for i in range(len(model_input['sub_dir']))]

            img_names = model_input['img_name'][:,0].cpu().numpy()
            print("Plotting images: {}".format(img_names))
            utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
            if eval_all:
                for dir in plot_dir:
                    utils.mkdir_ifnotexists(dir)
            plt.plot(img_names,
                     model_outputs,
                     ground_truth,
                     plot_dir,
                     self.start_epoch,
                     self.img_res,
                     is_eval=eval_all,
                     first=is_first_batch,
                     )
            is_first_batch = False
            del model_outputs, ground_truth

        if not self.plot_dataset.only_json:
            from utils.metrics import run as cal_metrics
            cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate')
            cal_metrics(output_dir=plot_dir[0], gt_dir=self.plot_dataset.gt_dir, pred_file_name='rgb_erode_dilate', no_cloth=True)



