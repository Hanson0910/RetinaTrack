from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils.prior_box import PriorBox
import cv2
from models.model.retinatrack import RetinaTrackNet
from config.config import cfg_re50
from utils.box_utils import decode
import time
import torchvision
import parser
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='RetinaTrack')
parser.add_argument('-m', '--trained_model', default='demo.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('-image', default='source/test.jpg', help='test image path')
args = parser.parse_args()

if __name__ == '__main__':
    cfg = cfg_re50
    model = RetinaTrackNet(cfg=cfg).cuda()
    param = torch.load('demo.pth', map_location=lambda storage, loc: storage.cuda('cuda:0'))
    model.load_state_dict(param)
    model.eval()
    img_raw = cv2.imread(args.image)
    img = cv2.resize(img_raw,(640,640))
    _,im_height,im_width= img.shape
    img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    img = img.cuda().float() / 255.0
    scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]]).cuda()
    tic = time.time()
    loc, conf, classifier = model(img)  # forward pass
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()
    
    with torch.no_grad():
        boxes = decode(loc.squeeze(0), priors, cfg['variance'])
        boxes = boxes * scale
        conf = F.softmax(conf, dim=-1)
        scores = conf.squeeze(0).cpu().numpy()[:, 1]
        inds = np.where(scores > 0.6)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        scores = torch.from_numpy(scores).cuda().unsqueeze(1)
        classifier = F.softmax(classifier, dim=-1).squeeze(0)
        classifier = classifier.data.max(-1, keepdim=True)[1]
        classifier = classifier[inds].float()
        dets = torch.cat((boxes,scores,classifier),1)
        i = torchvision.ops.boxes.nms(dets[:,:4], dets[:,5], args.nms_threshold)
        dets = dets[i]
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        text = "car: {:d}".format(int(b[5]))
        print(text)
        b = list(map(int, b))  
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imwrite('source/result_img.jpg', img_raw)