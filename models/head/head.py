import os
from collections import OrderedDict
import torch
import math
import torch.nn as nn
from models.neck.neck import conv_bn,conv_bn1X1
__all__ = ['RetinaTrackClassHead','RetinaTrackBBoxHead','RetinaTrackEmbedHead']

def task_specific_cls(inp, oup, stride = 1, leaky = 0, m2 = 1):
    peranchor_feature = nn.ModuleList()
    for j in range(m2):
        peranchor_feature.append(conv_bn(inp,oup,stride,leaky))
    nn.Sequential(*peranchor_feature)
    return nn.Sequential(*peranchor_feature)

def task_specific_loc(inp, oup, stride = 1, leaky = 0, m2 = 1):
    peranchor_feature = nn.ModuleList()
    for j in range(m2):
        peranchor_feature.append(conv_bn(inp,oup,stride,leaky))
    nn.Sequential(*peranchor_feature)
    return nn.Sequential(*peranchor_feature)

def task_specific_emb(inp, oup, stride = 1, leaky = 0, m3 = 2):
    peranchor_feature = nn.ModuleList()
    for j in range(m3):
        peranchor_feature.append(conv_bn1X1(inp,oup,stride,leaky))
    nn.Sequential(*peranchor_feature)
    return nn.Sequential(*peranchor_feature)

def make_cls_head(inp, oup = 2, fpnNum = 3, anchorNum = 2):
    cls_heads = nn.ModuleList()
    for i in range(fpnNum):
        for j in range(anchorNum):
            cls_heads.append(nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding=0))
    return cls_heads

def make_loc_head(inp, oup = 4, fpnNum = 3, anchorNum = 2):
    bbox_heads = nn.ModuleList()
    for i in range(fpnNum):
        for j in range(anchorNum):
            bbox_heads.append(nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding=0))
    return bbox_heads

def make_emb_head(inp, oup = 256, fpnNum = 3, anchorNum = 2):
    emb_heads = nn.ModuleList()
    for i in range(fpnNum):
        for j in range(anchorNum):
            emb_heads.append(nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding=0))
    return emb_heads