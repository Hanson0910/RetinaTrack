import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp,decode
import copy
import numpy as np
import torchvision

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,is_cuda = True):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.is_cuda = is_cuda
        self.idLoss = nn.CrossEntropyLoss(ignore_index=-1,reduction='sum')
        
    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds, classifier pred,
            priors: priors boxes from SSD net.
            targets (tensor): Ground truth boxes and labels for a batch,
            shape: [batch_size,num_objs,5] (batch_index,classification,xc,yc,w,h,ids).
        """

        loc_data, conf_data, classifier_data = predictions
        #print('&&&',loc_data.shape,classifier_data.shape)
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        classifier_t = torch.Tensor(num, num_priors)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            targets[idx][:, 1] = 1
            truths = targets[idx][:, 2:6].data
            labels = targets[idx][:, 1].data
            classifiers = targets[idx][:, 6].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, classifiers, loc_t, conf_t, classifier_t, idx)
        if self.is_cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            classifier_t = classifier_t.cuda()

        zeros = torch.tensor(0).cuda()
        # classifier Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(classifier_data)
        aa_p = classifier_data[pos_idx].view(-1, 7)
        classifier_t = classifier_t[pos].long()
        id_loss = self.idLoss(aa_p,classifier_t)
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        id_loss /= N
        
        return loss_l, loss_c, id_loss
