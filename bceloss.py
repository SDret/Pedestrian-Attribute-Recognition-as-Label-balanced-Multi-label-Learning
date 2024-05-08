import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import LOSSES
from tools.function import ratio2weight

@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.smoothing = None

        self.loss_mean = []
        self.label = []

        self.pos_loss_mean = 0
        self.neg_loss_mean = 0

    def forward(self, logits, targets, flag = True, epoch = 0):
        logits = logits[0]

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)
            loss_m = (loss_m * sample_weight.cuda())
                    
        if flag == True and epoch == 0:
            self.loss_mean.append(loss_m.detach().cpu().numpy())
            self.label.append(targets.cpu().numpy())

        if flag == True and epoch != 0:
            b = targets * self.pos_loss_mean.expand_as(targets) + (1-targets)*self.neg_loss_mean.expand_as(targets)
            loss_m = (loss_m - b).abs() + b

        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        return [loss], [loss_m]
