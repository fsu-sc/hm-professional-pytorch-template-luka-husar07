# %%
import argparse
import collections
import torch
import numpy as np
import data_loader.function_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.dynamic_model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import traceback
import sys
import shutil
import os
import glob
from pathlib import Path

def setup_directories():
    # Create necessary directories if they don't exist
    os.makedirs('outputs/logs', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)

    # Clean up models directory
    if os.path.exists('outputs/models'):
        shutil.rmtree('outputs/models')
        os.makedirs('outputs/models')

    # Clean up old logs but keep latest for each configuration
    log_dir = Path('outputs/logs')
    config_dirs = [d for d in log_dir.iterdir() if d.is_dir()]
    
    for config_dir in config_dirs:
        # Get all run directories for this configuration
        run_dirs = sorted([d for d in config_dir.iterdir() if d.is_dir()],
                         key=lambda x: x.stat().st_mtime)
        
        # Keep only the latest run
        if len(run_dirs) > 1:
            for old_run in run_dirs[:-1]:
                shutil.rmtree(old_run)

def main():
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # %% Read the args
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='configs/config.json', type=str,
                    help='config file path (default: configs/config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    # %%
    logger = config.get_logger('train')

    # setup data_loader instances
    try:
        data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.split_validation()
    except Exception as e:
        print("Error initializing data loader:")
        print(traceback.format_exc())
        sys.exit(1)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # %%
    trainer = Trainer(model, criterion, metrics, optimizer,
                    config=config,
                    device=device,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader,
                    lr_scheduler=lr_scheduler)

    trainer.train()

if __name__ == '__main__':
    setup_directories()
    main()