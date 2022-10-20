#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import math
import torchvision.models as models
import random
import numpy as np
import sys
from torch.nn import init

import numpy as np

sys.path.append('../../')
import utils
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

def weight_initialization_1(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()

class ResNet_DomainClassifier(nn.Module):
    
    def __init__(self, input_dim=256):
        super(ResNet_DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        self.bn1 =nn.BatchNorm1d(2)
        self.relu1 = nn.ReLU()
        # self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2, 5)
        self.bn2 =nn.BatchNorm1d(5)
        self.relu2 = nn.ReLU()
        # self.dout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(5, 2)
        self.prelu = nn.ReLU()
        self.out = nn.Linear(2, 2)
        self.cuda()
        self.apply(weight_initialization_1)
        # self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        a1 = self.bn1(a1)
        h1 = self.relu1(a1)
        # dout = self.dout(h1)
        a2 = self.fc2(h1)
        a2 = self.bn2(a2)
        h2 = self.relu2(a2)
        # dout2 = self.dout2(h2)
        a3 = self.fc3(h2)
        h3 = self.prelu(a3)
        a3 = self.out(h3)
        # y = self.out_act(a3)
        return a3


"""
Adapted from https://github.com/VisionLearningGroup/OVANet
"""
class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)


"""
Implements individual task networks
Adapted from https://github.com/jhoffman/cycada_release
"""
class TaskNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    "Basic class which does classification."
    def __init__(self, num_cls=8, weights_init=None):
        super(TaskNet, self).__init__()
        
        self.num_cls = num_cls        
        self.setup_net()        
        self.criterion = nn.CrossEntropyLoss()
        self.cuda()

    def forward(self, x, with_ft=False, with_emb=False, reverse_grad=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = x.clone()
        emb = self.fc_params(x)

        if isinstance(self.classifier, nn.Sequential): # LeNet
            emb = self.classifier[:-1](emb)
            if reverse_grad: emb = utils.ReverseLayerF.apply(emb)
            score = self.classifier[-1](emb)
        else:                                          # ResNet
            if reverse_grad: emb = utils.ReverseLayerF.apply(emb)
            score = self.classifier(emb)   
    
        if with_emb:
            return score, emb
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict, strict=False)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

class ResNet18(TaskNet):
    num_channels = 3
    name = 'ResNet18'

    def setup_net(self):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Identity()
        
        self.classifier = nn.Linear(512, self.num_cls)
        init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()


class ResNet34(TaskNet):
    num_channels = 3
    name = 'ResNet34'

    def setup_net(self):
        model = models.resnet34(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Identity()
        
        self.classifier = nn.Linear(512, self.num_cls)
        init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()


