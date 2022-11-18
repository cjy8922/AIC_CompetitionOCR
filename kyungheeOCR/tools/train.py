import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from vedastr.runners import TrainRunner
from vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('config', default='configs/small_satrn.py', help='config file path')
    parser.add_argument('gpus', default='0', help='target gpus')
    args = parser.parse_args()

    return args


def main():
    # Load Configuration Files
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    # Root Directory (save pth file)
    _, fullname = os.path.split(cfg_path)
    fname, ext = os.path.splitext(fullname)
    root_workdir = cfg.pop('root_workdir')
    workdir = os.path.join(root_workdir, fname)
    os.makedirs(workdir, exist_ok=True)

    # train_cfg, deploy_cfg, common_cfg
    train_cfg = cfg['train']
    deploy_cfg = cfg['deploy']
    common_cfg = cfg['common']
    common_cfg['workdir'] = workdir
    deploy_cfg['gpu_id'] = args.gpus.replace(" ", "")

    # Start Train
    runner = TrainRunner(train_cfg, deploy_cfg, common_cfg)
    runner()  # __call()__


if __name__ == '__main__':
    main()

