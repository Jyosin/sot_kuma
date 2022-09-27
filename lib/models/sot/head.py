''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: neck modules for SOT models
Data: 2021.6.23
'''

import torch
import torch.nn as nn
import numpy as np
from .modules import *

class Learn2Match(nn.Module):
    """
    target estimation head in "learn to match: Learn to Match: Automatic Ma tching Networks Design for Visual Tracking"
    https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learn_To_Match_Automatic_Matching_Network_Design_for_Visual_Tracking_ICCV_2021_paper.pdf
    """
    def __init__(self, in_channels=256, out_channels=256, roi_size=3):
        super(Learn2Match, self).__init__()
        # default parameters
        self.search_size = 255
        self.score_size = (self.search_size - 255) // 8 + 31
        self.batch = 32 if self.training else 1
        self.grids()

        # heads
        self.regression = L2Mregression(inchannels=in_channels, outchannels=out_channels, towernum=3)
        self.classification = L2Mclassification(roi_size=roi_size, stride=8.0, inchannels=in_channels)

    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.score_size
        stride = 8

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * stride + self.search_size // 2
        self.grid_to_search_y = y * stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)

        self.grid_to_search_x.requires_grad = False
        self.grid_to_search_y.requires_grad = False

    def pred_to_image(self, bbox_pred):
        if not bbox_pred.size(0) == self.batch:
            self.batch = bbox_pred.size(0)
            self.grids()

        if not self.score_size == bbox_pred.size(-1):
            self.score_size = bbox_pred.size(-1)
            self.grids()

        self.grid_to_search_x = self.grid_to_search_x.to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)  # 17*17
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)  # 17*17
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)  # 17*17
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)  # 17*17

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred

    def forward(self, inputs):
        xfs4, xfs3, zfs4, zfs3, template_mask, target_box,  = inputs['xf_conv4'], inputs['xf_conv3'], inputs['zf_conv4'], \
                                                              inputs['zf_conv3'], inputs['template_mask'], inputs['target_box']

        reg_outputs = self.regression(xf=xfs4, zf=zfs4, zfs3=zfs3, mask=template_mask, target_box=target_box)
        
        pred_box, target = self.pred_to_image(reg_outputs['reg_score']), [reg_outputs['zf_conv4'], reg_outputs['zf_conv3']]
        # self.pred_box = pred_box
        # self.target = target
        # import pdb
        # pdb.set_trace()
        if self.training:
            cls_label, jitterBox = inputs['cls_label'], inputs['jitterBox']
        else:
            cls_label, jitterBox = None, None
        cls_outputs = self.classification(pred_box, reg_outputs['reg_feature'], zfs4, xfs3, zfs3, target=target,
                                          mask=template_mask, cls_label=cls_label, jitterBox=jitterBox)

        return cls_outputs, reg_outputs
