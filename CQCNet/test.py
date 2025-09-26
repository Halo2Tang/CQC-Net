import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.qtunet import QTUNET
from models.unet2p.unet_2p import UNet_2Plus
from models.transfuse.TransFuse import TransFuse_L
from models.ulite import ULite
from ptflops import get_model_complexity_info
from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    test_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    model = QTUNET(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        # load_ckpt_path = model_cfg['load_ckpt_path']
    )

    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion.cuda()

    print('#----------Testing----------#')
    best_weight = torch.load('models/vmunet/best-epoch285-loss0.4245.pth', map_location=torch.device('cuda'))
    model.load_state_dict(best_weight)
    loss = test_one_epoch(
        test_loader,
        model,
        criterion,
        logger,
        config,
    )


if __name__ == '__main__':
    config = setting_config
    main(config)