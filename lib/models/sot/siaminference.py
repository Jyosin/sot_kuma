import torch
import importlib
import torch.nn as nn
import pdb

from lib.utils.box_helper import matcher
from lib.models.sot.modules import NestedTensor, nested_tensor_from_tensor, nested_tensor_from_tensor_2


class SiamInference(nn.Module):
    def __init__(self, archs=None):
        super(SiamInference, self).__init__()
        self.cfg = archs['cfg']
        self.init_arch(archs)
        self.init_hyper()
        self.init_loss()


    def init_arch(self, inputs):
        self.backbone = inputs['backbone']
        self.neck = inputs['neck']
        self.head = inputs['head']

    def init_hyper(self):
        self.lambda_u = 0.1
        self.lambda_s = 0.2
        # self.grids()

    def init_loss(self):
        if self.cfg is None:
            raise Exception('Not set config!')

        loss_module = importlib.import_module('models.sot.loss')

        cls_loss_type = self.cfg.MODEL.LOSS.CLS_LOSS
        reg_loss_type = self.cfg.MODEL.LOSS.REG_LOSS

        self.cls_loss = getattr(loss_module, cls_loss_type)
        cls_loss_add_type = self.cfg.MODEL.LOSS.CLS_LOSS_ADDITIONAL
        self.cls_loss_additional = getattr(loss_module, cls_loss_add_type)

        if reg_loss_type is None or reg_loss_type == 'None':
            pass
        else:
            self.reg_loss = getattr(loss_module, reg_loss_type)

    def forward(self, inputs):
        """
        inputs:
         - template: BCHW, H*W:127*127
         - search: BCHW, H*W:255*255
         - cls_label: BH'W' or B2H'W'
         - reg_label: B4H'W (optional)
         - reg_weight: BH'W' (optional)
        """

        template, search = inputs['template'], inputs['search']

        # backbone
        zfs = self.backbone(template)
        xfs = self.backbone(search)

        zf_conv4, zf_conv3 = zfs['p3'], zfs['p2']
        xf_conv4, xf_conv3 = xfs['p3'], xfs['p2']


        # neck
        zfs4, zfs3 = self.neck(zf_conv4, zf_conv3)
        xfs4, xfs3 = self.neck(xf_conv4, xf_conv3)

        # head
        # not implement Ocean object-aware version, if you need, pls find it in researchmm/TracKit

        head_inputs = {'xf_conv4': xfs4, 'xf_conv3': xfs3, 'zf_conv4': zfs4, 'zf_conv3': zfs3, \
                        'template_mask': inputs['template_mask'], 'target_box': inputs['template_bbox'],
                        'jitterBox': inputs['jitterBox'], 'cls_label': inputs['cls_label']
                        }
        cls_preds, reg_preds = self.head(head_inputs)

        cls_label, reg_label, reg_weight = inputs['cls_label'], inputs['reg_label'], inputs['reg_weight']
        reg_pred = reg_preds['reg_score']
        reg_loss = 2 * self.reg_loss(reg_pred, reg_label, reg_weight)
        cls_pred_s1, cls_pred_s2 = cls_preds['cls_score_s1'], cls_preds['cls_score_s2']
        cls_loss_s1 = self.cls_loss(cls_pred_s1, cls_label)
        cls_loss_s2 = self.cls_loss_additional(cls_pred_s2, cls_preds['cls_label_s2'], cls_preds['cls_jitter'], inputs['jitter_ious'])
        cls_loss = cls_loss_s1 + cls_loss_s2
        loss = {'cls_loss': cls_loss, 'reg_loss': reg_loss}

        return loss

    # only for testing
    def template(self, inputs):
        """
        inputs:
         - template: BCHW, H*W:127*127
         - template_mask: BHW (optional)
        """

        template = inputs['template']


        zfs = self.backbone(template)

        zf_conv4, zf_conv3 = zfs['p3'], zfs['p2']

        self.zfs4, self.zfs3 = self.neck(zf_conv4, zf_conv3)

        if 'template_mask' in inputs.keys():
            self.template_mask = inputs['template_mask'].float()

        if 'target_box' in inputs.keys():
            self.target_box = torch.tensor(inputs['target_box'], dtype=torch.float32).to(self.zfs3.device)
            self.target_box = self.target_box.view(1, 4)

    def track(self, inputs):
        """
        inputs:
         - search: BCHW, H*W:255*255
        """

        search = inputs['search']
        xfs = self.backbone(search)

        xf_conv4, xf_conv3 = xfs['p3'], xfs['p2']

        if self.neck is not None:
            xfs4, xfs3 = self.neck(xf_conv4, xf_conv3)
            head_inputs = {'xf_conv4': xfs4, 'xf_conv3': xfs3, 'zf_conv4': self.zfs4, 'zf_conv3': self.zfs3, \
                            'template_mask': self.template_mask, 'target_box': self.target_box, }
            # pdb.set_trace()
            cls_preds, reg_preds = self.head(head_inputs)
            preds = {
                'cls_s1': cls_preds['cls_score_s1'],
                'cls_s2': cls_preds['cls_score_s2'],
                'reg': reg_preds['reg_score'] # clip large regression pred
            }

            # record some feats for zoom
            self.record = [cls_preds['xf_conv4'].detach(), cls_preds['xf_conv3'].detach(),
                            cls_preds['zf_conv4'].detach(), cls_preds['zf_conv3'].detach()]  # [xf_conv4, xf_conv3, zf_conv4, zf_conv3]

        else:
            preds = self.head(xf, self.zf)

        if 'reg' not in preds.keys():
            preds['reg'] = None

        return preds

    def zoom(self, box):
        """
        zoom trick in AutoMatch
        """
        cls_pred = self.head.classification(None, self.record[0], self.record[2], self.record[1], self.record[3], zoom_box=box)

        return cls_pred.squeeze()













