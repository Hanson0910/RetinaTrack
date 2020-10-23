import os
from collections import OrderedDict
import torch
import math
import torch.nn as nn
from models.neck.neck import conv_bn,conv_bn1X1
__all__ = ['RetinaTrackClassHead','RetinaTrackBBoxHead','RetinaTrackEmbedHead']

def task_specific_cls(inp, oup, stride = 1, leaky = 0, m2 = 1):
    peranchor_feature = []
    for j in range(m2):
        peranchor_feature.append(conv_bn(inp,oup,stride,leaky))
    nn.Sequential(*peranchor_feature)
    return nn.Sequential(*peranchor_feature)

def task_specific_loc(inp, oup, stride = 1, leaky = 0, m2 = 1):
    peranchor_feature = []
    for j in range(m2):
        peranchor_feature.append(conv_bn(inp,oup,stride,leaky))
    nn.Sequential(*peranchor_feature)
    return nn.Sequential(*peranchor_feature)

def task_specific_emb(inp, oup, stride = 1, leaky = 0, m3 = 2):
    peranchor_feature = []
    for j in range(m3):
        peranchor_feature.append(conv_bn1X1(inp,oup,stride,leaky))
    nn.Sequential(*peranchor_feature)
    return nn.Sequential(*peranchor_feature)

def make_cls_head(inp, oup = 2, numstage = 3, numanchor = 3):
    cls_heads = []
    for i in range(numstage):
        for j in range(numanchor):
            cls_heads.append(nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding=0))
    return cls_heads

def make_loc_head(inp, oup = 4, numstage = 3, numanchor = 3):
    bbox_heads = []
    for i in range(numstage):
        for j in range(numanchor):
            bbox_heads.append(nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding=0))
    return bbox_heads

def make_emb_head(inp, oup = 256, numstage = 3, numanchor = 3):
    emb_heads = []
    for i in range(numstage):
        for j in range(numanchor):
            emb_heads.append(nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding=0))
    return emb_heads