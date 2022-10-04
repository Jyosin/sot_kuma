''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: bbox manipulation and calculation (e.g. iou)
Data: 2021.6.23
'''

import math
import torch
import numpy as np
from shapely.geometry import Polygon, box
from collections import namedtuple

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """
    [x1, y1, x2, y2] --> [cx, cy, w, h]
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h

def center2corner(center):
    """
    [cx, cy, w, h] --> [x1, y1, x2, y2]
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2

def IoU(rect1, rect2):
    """
    calculation overlap between boxes
    input: [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)

    target_a = (tx2-tx1) * (ty2 - ty1)

    inter = ww * hh
    overlap = inter / (area + target_a - inter)

    return overlap

def matcher(reg_pred, search_bbox):
    """ Performs the matching
    Returns:
        A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order),
              and it is always 0, because single target tracking has only one target per image
        For each batch element, it holds:
            len(index_i) = len(index_j)
    """
    indices = []
    bs, num_queries = reg_pred.shape[:2]
    for i in range(bs):
        xmin, ymin, xmax, ymax = search_bbox[i]
        xmin = xmin.item()
        ymin = ymin.item()
        xmax = xmax.item()
        ymax = ymax.item()
        len_feature = int(np.sqrt(num_queries))
        Xmin = int(np.ceil(xmin*len_feature))
        Ymin = int(np.ceil(ymin*len_feature))
        Xmax = int(np.ceil(xmax*len_feature))
        Ymax = int(np.ceil(ymax*len_feature))
        if Xmin == Xmax:
            Xmax = Xmax+1
        if Ymin == Ymax:
            Ymax = Ymax+1
        a = np.arange(0, num_queries, 1)
        b = a.reshape([len_feature, len_feature])
        c = b[Ymin:Ymax,Xmin:Xmax].flatten()
        d = np.zeros(len(c), dtype=int)
        indice = (c,d)
        indices.append(indice)
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

