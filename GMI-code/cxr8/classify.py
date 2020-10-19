# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math, evolve

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class IR50(nn.Module):
    def __init__(self, num_classes=7):
        super(IR50, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.k = self.feat_dim // 2
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.num_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out

# eval
class VGG19(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG19, self).__init__()
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.feature = model.features
        self.feat_dim = 2048
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
        self.model = model
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        # import pdb; pdb.set_trace()
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        # import pdb; pdb.set_trace()
        
        return res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return res

class VGG16(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.feature = model.features
        self.feat_dim = 2048
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
        self.model = model
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        # import pdb; pdb.set_trace()
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        # import pdb; pdb.set_trace()
        
        return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return res

# target
class IR18(nn.Module):
    def __init__(self, num_classes=7):
        super(IR18, self).__init__()
        model = torchvision.models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feat_dim = 512
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        model.fc = nn.Linear(self.feat_dim, self.num_classes)
        self.model = model

    def forward(self, x):
        # print(self.model)
        # import pdb; pdb.set_trace()
        # feature = self.feature(x)
        # feature = feature.view(feature.size(0), -1)
        # import pdb; pdb.set_trace()
        # feature = self.bn(feature)
        # res = self.fc_layer(feature)
        res = self.model(x)

        return res

class mobilenet_v2(nn.Module):
    def __init__(self, num_classes=7):
        super(mobilenet_v2, self).__init__()
        model = torchvision.models.mobilenet_v2(pretrained=False)
        model.features[0] = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.feat_dim = 5120
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
        self.feature = model.features

    def forward(self, x):
        # print(self.model)
        
        # feature = self.feature(x)
        # feature = feature.view(feature.size(0), -1)
        # import pdb; pdb.set_trace()
        # feature = self.bn(feature)
        # res = self.fc_layer(feature)
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        # import pdb; pdb.set_trace()
        feature = self.bn(feature)
        res = self.fc_layer(feature)


        return res

        

class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
    