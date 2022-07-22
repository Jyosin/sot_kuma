import math
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

def build_simple_siamese_opt_lr(cfg, trainable_params):
    '''
    simple learning rate scheduel, used in SiamFC and SiamDW
    '''
    optimizer = torch.optim.SGD(trainable_params, cfg.TRAIN.LR,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = np.logspace(math.log10(cfg.TRAIN.LR), math.log10(cfg.TRAIN.LR_END),
                            cfg.TRAIN.END_EPOCH)

    return optimizer, lr_scheduler