'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.networks import Net

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        #self.conv1 = nn.Conv1d(in_planes, 4*growth_rate, kernel_size=1, padding=0, bias=False)
        #self.bn2 = nn.BatchNorm1d(4*growth_rate)
        self.conv2 = nn.Conv1d(in_planes, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        #out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn1(x)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm1d(in_planes)
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool1d(out, 3)
        return out


class DenseNet(Net):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        #super(DenseNet, self).__init__()
        super().__init__()
        self.growth_rate = growth_rate
        self.convs = nn.ModuleList()

        num_planes = 2*growth_rate
        self.convs.append(nn.Sequential(nn.Conv1d(in_channels=1, out_channels=growth_rate, kernel_size=31, stride=1, padding=15, bias=False),
                                        nn.MaxPool1d(kernel_size = 2),
                                        nn.BatchNorm1d(growth_rate),
                                        nn.ReLU(growth_rate),
                                        nn.Conv1d(in_channels=growth_rate, out_channels=growth_rate*2, kernel_size=13, stride=1, padding=6, bias=False),
                                        nn.MaxPool1d(kernel_size = 2),
                                        ))
        for idx in range(len(nblocks)):
            conv = []
            conv.append(self._make_dense_layers(block, num_planes, nblocks[idx]))
            num_planes += nblocks[idx]*growth_rate
            if not idx == len(nblocks) - 1:
                out_planes = int(math.floor(num_planes*reduction))
                conv.append(Transition(num_planes, out_planes))
                num_planes = out_planes
            self.convs.append(nn.Sequential(*conv))

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
        
        self.fine_tune_layers = nn.ModuleList()
        self.fine_tune_layers += self.convs[2:]
        self.fine_tune_layers.append(self.bn)
        self.fine_tune_layers.append(self.linear)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, all_tensors = False, give_features = False):
        if all_tensors:
            out_tensors = []
            out_tensors.append(x)
            
        for conv in self.convs:
            x = conv(x)
            if all_tensors:
                out_tensors.append(x)
            #print(x.shape)
        x = F.relu(self.bn(x))
        #x = F.avg_pool1d(x, 18)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        if give_features:
            return x
        if all_tensors:
            out_tensors.append(x)
        x = self.linear(x)
        
        if all_tensors:
            out_tensors.append(x)
            return out_tensors
        x = F.log_softmax(x, dim=1)
        return x

def DenseNet_vib(num_classes):
    return DenseNet(Bottleneck, [6,10,12,8], growth_rate=6, num_classes = num_classes)

def test():
    net = DenseNet_vib()
    x = torch.randn(1,1,2048)
    y = net(x)
    print(y)

# test()
