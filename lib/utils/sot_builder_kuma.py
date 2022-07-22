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

        return head