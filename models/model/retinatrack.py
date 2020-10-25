import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from models.neck.neck import FPN as FPN
from models.neck.neck import SSH as SSH
from models.neck.neck import task_shared
from models.head.head import task_specific_cls,task_specific_loc,task_specific_emb
from models.head.head import make_cls_head,make_loc_head,make_emb_head
from config.config import cfg_re50

class RetinaTrackNet(nn.Module):

    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaTrackNet, self).__init__()
        self.phase = phase
        backbone = None    
            
        if cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        else:
            pass
        
        anchorNum = cfg['anchorNum_per_stage'] ##not total anchor,indicate per stage anchors
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        ##task shared
        self.task_shared = task_shared(out_channels,out_channels,anchorNum = anchorNum)
        ##task specific
        self.cls_task = task_specific_cls(out_channels,out_channels)
        self.loc_task = task_specific_loc(out_channels,out_channels)
        self.emb_task = task_specific_emb(out_channels,out_channels)
        ###head
        self.cls_heads = make_cls_head(inp=out_channels,fpnNum=len(in_channels_list),anchorNum=anchorNum)
        self.loc_heads = make_loc_head(inp=out_channels,fpnNum=len(in_channels_list),anchorNum=anchorNum)
        self.emb_heads = make_emb_head(inp=out_channels,fpnNum=len(in_channels_list),anchorNum=anchorNum)

        ###classifier ,ndim of emb dims and num of ids
        self.classifier = nn.Linear(256, 7)


    def forward(self, inputs):
        out = self.body(inputs)
        # FPN
        fpn = self.fpn(out)
        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        fpnfeatures = [feature1, feature2, feature3]
        features = []
        ##task shared
        for fpnfeature in fpnfeatures:
            per_anchor_feature = []
            for per_task_shared in self.task_shared:
                per_anchor_feature.append(per_task_shared(fpnfeature))
            features.append(per_anchor_feature)
        ##task specific
        cls_heads,loc_heads,emb_heads = [],[],[]
        for i, per_fpn_features in enumerate(features):
            for j, per_anchor_feature in enumerate(per_fpn_features):
                cls_task_feature = self.cls_task(per_anchor_feature)
                loc_task_feature = self.loc_task(per_anchor_feature)
                emb_task_feature = self.emb_task(per_anchor_feature)
                ##cls feature,only one class but with background total class is two
                cls_head = self.cls_heads[i * len(per_fpn_features) + j](cls_task_feature)
                cls_head = cls_head.permute(0, 2, 3, 1).contiguous().view(cls_head.shape[0], -1, 2)
                ##loc frature,(x,y,w,h)
                loc_head = self.loc_heads[i * len(per_fpn_features) + j](loc_task_feature)
                loc_head = loc_head.permute(0, 2, 3, 1).contiguous().view(loc_head.shape[0], -1, 4)
                ##emb feature with 256 dim
                emb_head = self.emb_heads[i * len(per_fpn_features) + j](emb_task_feature)
                emb_head = emb_head.permute(0, 2, 3, 1).contiguous().view(emb_head.shape[0], -1, 256)
                
                cls_heads.append(cls_head)
                loc_heads.append(loc_head)
                emb_heads.append(emb_head)
        
        bbox_regressions = torch.cat([feature for i, feature in enumerate(loc_heads)], dim=1)
        classifications = torch.cat([feature for i, feature in enumerate(cls_heads)], dim=1)
        emb_features = torch.cat([feature for i, feature in enumerate(emb_heads)], dim=1)
        classifier = self.classifier(emb_features)
        return [bbox_regressions,classifications,classifier]

if __name__ == '__main__':
    cfg = cfg_re50
    model = RetinaTrackNet(cfg=cfg)
    inpunt = torch.randn(1, 3, 224, 224)
    cls_heads,loc_heads,emb_heads = model(inpunt)