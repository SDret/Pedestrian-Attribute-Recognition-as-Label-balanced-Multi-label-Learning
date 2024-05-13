import math

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange,repeat,reduce
from torch.nn.parameter import Parameter
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from tools.function import attributes_Q

def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)

class Classifier(nn.Module):

    def __init__(self, c_in, nattr, bb):
        super(Classifier, self).__init__()

        self.bb = bb
        self.nattr = nattr
        self.c = c_in
        self.attri_Q = attributes_Q(self.nattr, self.c)

        self.separte = nn.Sequential(nn.Linear(c_in, nattr*c_in), nn.BatchNorm1d(nattr*c_in))
        self.logits = nn.ModuleList([nn.Linear(c_in, 1) for i in range(nattr)])

        for p in self.separte.parameters():
            p.requires_grad=False

        for p in self.logits.parameters():
            p.requires_grad=False

    def forward(self, x, label=None, mode='train', epoch = 0):
        
        if self.bb == 'resnet50':
            x = rearrange(x, 'n c h w ->n (h w) c')
            x = reduce(x,'n k c ->n c', reduction = 'mean')
        
        x = self.separte(x)
        x = torch.reshape(x,[x.shape[0], self.nattr, self.c])

        if mode == 'train':
            self.attri_Q.update(x, label)

        logits = []
        for i in range(self.nattr):
            logits.append(self.logits[i](F.relu(x[:,i,:].squeeze())))
        logits = torch.cat(logits,dim=1)

        return logits, label, x
    
class Network(nn.Module):
    def __init__(self, backbone, classifier, number):
        super(Network, self).__init__()

        self.number = number
        self.backbone = backbone
        self.classifier = classifier
        self.logits_ft = nn.ModuleList([nn.Linear(self.classifier.c, 1) for i in range(self.classifier.nattr)])

        for p in self.backbone.parameters():
            p.requires_grad=False

    def forward(self, x, label=None, mode='train', epoch = 0):
        
        x = self.backbone(x)
        logits, label, x = self.classifier(x,label,mode, epoch)

        if mode == 'train':
        
            if epoch > self.number:
                x_ft, label_ft = self.classifier.attri_Q.pop(x.shape[0])
                x_ft = x_ft.detach()
            else:
                x_ft = x.detach()
                label_ft = label

            logits_ft = []
            for i in range(self.classifier.nattr):
                logits_ft.append(self.logits_ft[i](F.relu(x_ft[:,i,:].squeeze())))
            logits_ft = torch.cat(logits_ft,dim=1)

            return [logits], [logits_ft], label, label_ft
        else:
            logits_ft = []
            for i in range(self.classifier.nattr):
                logits_ft.append(self.logits_ft[i](F.relu(x[:,i,:].squeeze())))
            logits_ft = torch.cat(logits_ft,dim=1)

            return [logits_ft],label
