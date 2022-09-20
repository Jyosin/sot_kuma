import importlib
import torch.nn as nn
from lib.models.sot.siaminference import SiamInference

class Siamese_builder(nn.Module):
    def __init__(self, cfg):
        super(Siamese_builder).__init__()
        self.cfg = cfg
        self.backbone = None
        self.neck = None
        self.head = None

    def build(self):
        backbone_type = self.cfg.MODEL.BACKBONE.NAME
        neck_type = self.cfg.MODEL.NECK.NAME
        head_type = self.cfg.MODEL.HEAD.NAME

        # backbone
        print('model backbone: {}'.format(backbone_type))
        backbone = self.build_backbone(backbone_type)

        # neck
        print('model neck: {}'.format(neck_type))
        neck = self.build_neck(neck_type)

        # head
        print('model head: {}'.format(head_type))
        head = self.build_head(head_type)

        print('model build done!')

        inputs = {'backbone': backbone, 'neck': neck, 'head': head, 'cfg': self.cfg}
        return SiamInference(archs=inputs)

    def build_backbone(self, backbone_type):
        bk_module = importlib.import_module('models.backbone.ResNet')
        bk_func = getattr(bk_module, backbone_type)
        backbone = bk_func(used_layers=self.cfg.MODEL.BACKBONE.LAYER)
    
        return backbone

    def build_neck(self, neck_type):
        if neck_type is None or neck_type == 'None':
            return None

        neck_module = importlib.import_module('models.sot.neck')
        neck_func = getattr(neck_module, neck_type)
        neck = neck_func(in_channels=self.cfg.MODEL.NECK.IN_CHANNEL, out_channels=self.cfg.MODEL.NECK.OUT_CHANNEL)

        return neck

    def build_head(self, head_type):
        head_module = importlib.import_module('models.sot.head')
        head_func = getattr(head_module, head_type)
        head = head_func(in_channels=self.cfg.MODEL.HEAD.IN_CHANNEL,
                        out_channels=self.cfg.MODEL.HEAD.IN_CHANNEL)

        return head
