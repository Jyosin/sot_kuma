''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: set learning rate for training
Data: 2021.6.23
'''

import math
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


class LRScheduler(_LRScheduler):
    '''
    super class for learning rate schedule
    '''
    def __init__(self, optimizer, last_epoch=-1):
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr
                for pg in self.optimizer.param_groups]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(self.__class__.__name__,
                                             self.lr_spaces)


class LogScheduler(LRScheduler):
    '''
    learning rate in log space
    '''
    def __init__(self, optimizer, start_lr=0.03, end_lr=5e-4,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        self.lr_spaces = np.logspace(math.log10(start_lr),
                                     math.log10(end_lr),
                                     epochs)

        super(LogScheduler, self).__init__(optimizer, last_epoch)


class StepScheduler(LRScheduler):
    '''
    learning rate in step decreasing
    '''
    def __init__(self, optimizer, start_lr=0.01, end_lr=None,
                 step=10, mult=0.1, epochs=50, last_epoch=-1, **kwargs):
        if end_lr is not None and end_lr != 'None':
            if start_lr is None:
                start_lr = end_lr / (mult ** (epochs // step))
            else:  # for warm up policy
                mult = math.pow(end_lr/start_lr, 1. / (epochs // step))
        self.start_lr = start_lr
        self.lr_spaces = self.start_lr * (mult**(np.arange(epochs) // step))
        self.mult = mult
        self._step = step

        super(StepScheduler, self).__init__(optimizer, last_epoch)


class WarmUPScheduler(LRScheduler):
    '''
    warmup strategy
    '''
    def __init__(self, optimizer, warmup, normal, epochs=50, last_epoch=-1):
        warmup = warmup.lr_spaces  # [::-1]
        normal = normal.lr_spaces
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmUPScheduler, self).__init__(optimizer, last_epoch)

LRs = {
    'log': LogScheduler,
    'step': StepScheduler,}


def _build_lr_scheduler(optimizer, config, epochs=50, last_epoch=-1):
    return LRs[config.TYPE](optimizer, last_epoch=last_epoch, epochs=epochs, **config.KWARGS)


def _build_warm_up_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):
    warmup_epoch = cfg.TRAIN.WARMUP.EPOCH
    sc1 = _build_lr_scheduler(optimizer, cfg.TRAIN.WARMUP, warmup_epoch, last_epoch)
    sc2 = _build_lr_scheduler(optimizer, cfg.TRAIN.LR, epochs - warmup_epoch, last_epoch)
    return WarmUPScheduler(optimizer, sc1, sc2, epochs, last_epoch)


def build_lr_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):
    '''
    build learning rate schedule
    '''
    if cfg.TRAIN.WARMUP.IFNOT:
        return _build_warm_up_scheduler(optimizer, cfg, epochs, last_epoch)
    else:
        return _build_lr_scheduler(optimizer, cfg.TRAIN.LR, epochs, last_epoch)


def build_siamese_opt_lr(cfg, model, current_epoch=0):
    '''
    common learning tricks in Siamese: fix backbone (warmup) --> unfix
    '''
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            m.eval()

    if current_epoch >= cfg.TRAIN.UNFIX_EPOCH:
        if len(cfg.TRAIN.TRAINABLE_LAYER) > 0:  # specific trainable layers
            for layer in cfg.TRAIN.TRAINABLE_LAYER:
                for param in getattr(model.backbone, layer).parameters():
                    param.requires_grad = True
                for m in getattr(model.backbone, layer).modules():
                    if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                        m.train()
        else:    # train all backbone layers
            for param in model.backbone.parameters():
                param.requires_grad = True
            for m in model.backbone.modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                    m.train()
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for m in model.backbone.modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad, model.backbone.parameters()),
                          'lr': cfg.TRAIN.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.neck.parameters(),
                        'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, cfg, epochs=cfg.TRAIN.END_EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler
