from models.model.retinatrack import RetinaTrackNet
from config.config import cfg_re50
import torch

if __name__ == '__main__':
    cfg = cfg_re50
    model = RetinaTrackNet(cfg=cfg)
    inpunt = torch.randn(1, 3, 224, 224)
    cls_heads,loc_heads,emb_heads = model(inpunt)
    print('Over!!')