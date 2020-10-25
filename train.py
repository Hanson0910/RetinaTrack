from models.model.retinatrack import RetinaTrackNet
from config.config import cfg_re50
import torch
from utils.prior_box import PriorBox
from utils.multibox_loss import MultiBoxLoss
import cv2
import numpy as np
import torch.optim as optim

if __name__ == '__main__':
    cfg = cfg_re50
    model = RetinaTrackNet(cfg=cfg).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(2, 0.35, True, 0, True, 7, 0.35, False)
    img = cv2.imread('source/test.jpg')
    img = cv2.resize(img,(640,640))
    targets = np.loadtxt('source/label.txt')
    targets = np.expand_dims(targets,0)
    targets = torch.from_numpy(targets).float().cuda()
    inpunt = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    inpunt = np.ascontiguousarray(inpunt)
    inpunt = torch.from_numpy(inpunt).unsqueeze(0).cuda()
    inpunt = inpunt.cuda().float() / 255.0
    while True:
        outputs = model(inpunt)
        priorbox = PriorBox(cfg)
        with torch.no_grad():
            priors = priorbox.forward()
            priors = priors.cuda()
        loss_l, loss_c, loss_id = criterion(outputs, priors, targets)
        loss = loss_l + loss_c + loss_id
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < 0.02:
            torch.save(model.state_dict(),'demo.pth')
            break
        print('total loss: ',loss.item())