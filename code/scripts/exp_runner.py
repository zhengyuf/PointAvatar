import sys
sys.path.append('../code')
sys.path.append('./')
import argparse

from scripts.train import TrainRunner
from scripts.test import TestRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument('--is_eval', default=False, action="store_true", help='If set, only render images')
    # Training flags
    parser.add_argument('--nepoch', type=int, default=62, help='number of epochs to train for')
    parser.add_argument('--wandb_workspace', type=str)
    parser.add_argument('--wandb_tags', type=str, nargs="+", default=[])
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.')
    # Testing flags
    parser.add_argument('--only_json', default=False, action="store_true", help='If set, do not load images during testing. ')
    # Checkpoints
    parser.add_argument('--checkpoint', default='latest', type=str, help='The checkpoint epoch number in case of continuing from a previous run.')
    parser.add_argument('--load_path', type=str, default='', help='If set explicitly, then load from this path, instead of the continue-scripts path')
    opt = parser.parse_args()

    if not opt.is_eval:
        runner = TrainRunner(conf=opt.conf,
                             nepochs=opt.nepoch,
                             checkpoint=opt.checkpoint,
                             is_continue=opt.is_continue,
                             load_path=opt.load_path,
                             wandb_workspace=opt.wandb_workspace,
                             wandb_tags=opt.wandb_tags,
                             )
        runner.run()

        runner = TestRunner(conf=opt.conf,
                            checkpoint='latest',
                            load_path=opt.load_path,
                            only_json=opt.only_json,
                            )

        runner.run()
    else:
        runner = TestRunner(conf=opt.conf,
                            checkpoint=opt.checkpoint,
                            load_path=opt.load_path,
                            only_json=opt.only_json,
                            )

        runner.run()
