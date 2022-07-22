import _init_paths
import os
import torch
import argparse
from easydict import EasyDict as edict
import utils.read_file_kuma as reader
import utils.lr_scheduler_kuma as learner
import utils.sot_builder_kuma as builder

from torch.nn import DataParallel
from torch.utils.data import DataLoader
from dataset.siamese_builder_kuma import SiameseDataset as data_builder
from core.trainer.siamese_train_kuma import siamese_train as trainer
eps = 1e-5

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train Ocean')
    parser.add_argument('--cfg', type=str, default='experiments/AutoMatch.yaml', help='yaml configure file name')
    parser.add_argument('--wandb', action='store_true', help='use wandb to watch training')
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    return args

def epoch_train(config, logger=None, writer_dict=None, wandb_instance=None, args=None):
    # create model
    print('====> build model <====')
   
    siambuilder = builder.Siamese_builder(config)
    model = siambuilder.build()
    model = model.cuda()

    start_epoch = config.TRAIN.START_EPOCH 
    optimizer, lr_scheduler = learner.build_siamese_opt_lr(config, model, config.TRAIN.START_EPOCH)

    # create parallel
    gpus = [int(i) for i in config.COMMON.GPUS.split(',')]
    gpu_num = world_size = len(gpus)  # or use world_size = torch.cuda.device_count()
    gpus = list(range(0, gpu_num))

    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    model = DataParallel(model, device_ids=gpus).to(device)

    # train
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        train_set = data_builder(config)
        train_loader = DataLoader(train_set, batch_size=config.TRAIN.BATCH * gpu_num, num_workers=config.TRAIN.WORKERS,
                            pin_memory=True, sampler=None, drop_last=True)
        lr_scheduler.step(epoch)
        curLR = lr_scheduler.get_cur_lr()

        inputs = {'data_loader': train_loader, 'model': model, 'optimizer': optimizer, 'device': device,
                'epoch': epoch + 1, 'cur_lr': curLR, 'config': config,
                'writer_dict': writer_dict, 'logger': logger, 'wandb_instance': wandb_instance}
        model, _ = trainer(inputs)


def main():
    # read config
    print('====> load configs <====')
    args = parse_args()
    config = edict(reader.load_yaml(args.cfg))
    os.environ['CUDA_VISIBLE_DEVICES'] = config.COMMON.GPUS

    epoch_train(config, logger=None, writer_dict=None, wandb_instance=None, args=args)

if __name__ == '__main__':
    main()