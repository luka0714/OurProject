
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math

import torchvision
from deform_conv_v2 import DeformConv2d

# conv + batchnorm + relu
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True)
    )
    return layer

class MyInception(nn.Module):
    def __init__(self, in_channel, out1, out2, kernel1, kernel2, padding1, padding2,):
        super(MyInception, self).__init__()
        #self.branch1x1 = conv_relu(in_channel, out1, kernel1, padding = padding1) # 5*5 padding = 2   3*3 padding = 1
        self.branch1x1 = nn.Conv2d(in_channels = in_channel, out_channels = out1, kernel_size = kernel1, padding = padding1)
        self.branch1x2 = nn.Conv2d(in_channels = in_channel, out_channels = out2, kernel_size = kernel2, padding = padding2)
        #self.branch1x2 = conv_relu(in_channel, out2, kernel2, padding = padding2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        f1 = self.relu(self.branch1x1(x))
        f2 = self.relu(self.branch1x2(x))
        output = torch.cat((f1,f2),dim = 1)
        return output 

class MyexInception(nn.Module):
    def __init__(self, in_channel, out1, out2, out3, out4, out5, out6, kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, 
                padding1, padding2, padding3, padding4, padding5, padding6):
        super(MyexInception, self).__init__()
        self.branch1x1 = conv_relu(in_channel, out1, kernel1, padding = padding1) # 5*5 padding = 2   3*3 padding = 1

        self.branch1x2 = conv_relu(in_channel, out2, kernel2, padding = padding2)

        self.branch1x3 = conv_relu(in_channel, out3, kernel3, padding = padding3)

        self.branch1x4 = conv_relu(in_channel, out4, kernel4, padding = padding4)
        self.branch1x5 = conv_relu(in_channel, out5, kernel5, padding = padding5)
        self.branch1x6 = conv_relu(in_channel, out6, kernel6, padding = padding6)
    def forward(self,x):
        f1 = self.branch1x1(x)
        f2 = self.branch1x2(x)
        f3 = self.branch1x3(x)
        f4 = self.branch1x4(x)
        f5 = self.branch1x5(x)
        f6 = self.branch1x6(x)
        output = torch.cat((f1,f2,f3,f4,f5,f6),dim = 1)
        return output 

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        # 顺序： BN-relu-conv1(1*1)-BN-relu-conv2(3*3)
        out = self.conv1(F.relu(self.bn1(x)))  
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        # 顺序： BN-relu-conv1(3*3)
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        # 顺序： BN-relu-conv1(1*1)
        out = self.conv1(F.relu(self.bn1(x)))
        # out = F.avg_pool2d(out, 2) 
        return out

# growthRate:每一个Bottleneck之后增加多少 （growthRate设置为12）
class DenseNet(nn.Module):
    def __init__(self, growthRate, reduction, bottleneck):
        super(DenseNet, self).__init__()

        # nDenseBlocks：一个denseblock块里有多少层，假设4层
        nDenseBlocks = 4

        nChannels = 2*growthRate   # nChannels=24
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        # 一个dense block输出的feature map（即out channel）是dense层个feature map（即growthRate）相加
        nChannels += nDenseBlocks*growthRate  
        nOutChannels = int(math.floor(nChannels*reduction))  # reduction是过渡层的压缩系数，为了进一步提高模型的紧凑性
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans3 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans4 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense5 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv2 = nn.Conv2d(nChannels, 1, kernel_size=3, padding=1, bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate  # 在一个denseblock中，每有一个dense层输出就得与前面dense层输出的feature map数叠加
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.trans4(self.dense4(out))
        out = self.dense5(out)
        out = self.conv2(F.relu(self.bn1(out)))
        return out

# net1.0
class MyNet(nn.Module):
    def __init__(self, channels):  
        super(MyNet, self).__init__()
        layers = []
        
        # layers1 
        layers.append(conv_relu(in_channel = channels, out_channel = 64, kernel = 5, padding = 2))
        # layers2 不同尺寸滤波器技术
        layers.append(MyInception(64, 16, 32, 5, 3, 2, 1))
        # layers3 不同尺寸滤波器技术
        layers.append(MyInception(48, 16, 32, 3, 1, 1, 0))
        # layers4 最后一层论文中没有relu
        # layers.append(conv_relu(in_channel = 48, out_channel = 1, kernel = 3, padding = 1))
        layers.append(nn.Conv2d(in_channels = 48, out_channels = 1, kernel_size = 3, stride = 1, padding = 1))
        # 模块将按照在构造函数中传递的顺序添加到模块中。或者，也可以传递模块的有序字典。
        self.MyNet = nn.Sequential(*layers)
        
    #定义自己的前向传播函数
    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        out = self.MyNet(x)
        return out


class MergeNet(nn.Module):
    def __init__(self, channels, growthRate, reduction, bottleneck):  
        super(MergeNet, self).__init__()
        
        nDenseBlocks = 4  
        self.cvre1 = conv_relu(in_channel = channels, out_channel = 64, kernel = 5, padding = 2)
        self.incp1 = MyInception(64, 16, 32, 5, 3, 2, 1)
        self.incp2 = MyInception(48, 16, 32, 3, 1, 1, 0)
        
        nChannels = 48
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        # 一个dense block输出的feature map（即out channel）是dense层个feature map（即growthRate）相加
        nChannels += nDenseBlocks*growthRate  
        nOutChannels = int(math.floor(nChannels*reduction))  # reduction是过渡层的压缩系数，为了进一步提高模型的紧凑性
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, 1, kernel_size=3, padding=1, bias=False)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate  # 在一个denseblock中，每有一个dense层输出就得与前面dense层输出的feature map数叠加
        return nn.Sequential(*layers)

    #定义自己的前向传播函数
    def forward(self, x):
        # x = x.type(torch.cuda.FloatTensor)
        out = self.cvre1(x)
        out = self.incp1(out)
        out = self.incp2(out)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.conv1(F.relu(self.bn1(out)))
        return out


# ref-net
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1,map_reduce=6):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride, bn=False),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=1,relu=False, bn=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=False),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1), bn=False),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False, bn=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=False),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1, bn=False),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1, bn=False),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False, bn=False)
                )
        # self.branch3 = nn.Sequential(
        #         BasicConv(in_planes, inter_planes, kernel_size=1, stride=1,bn=False),
        #         BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1,bn=False),
        #         BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1,bn=False),
        #         BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=7, dilation=7, relu=False,bn=False)
        #         )

        self.ConvLinear = BasicConv(map_reduce*inter_planes, out_planes, kernel_size=1, stride=1, relu=False,bn=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False,bn=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # x3 = self.branch3(x)

        # out = torch.cat((x0,x1,x2,x3),1)
        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        shortCut = self.shortcut(x)
        out = out*self.scale + shortCut
        out = self.relu(out)

        return out

class BasicRFB_S(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1,map_reduce=8):
        super(BasicRFB_S, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,5), stride=1, padding=(0,2)),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=(5,1), stride=stride, padding=(2,0)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(8*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class RFNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(RFNet, self).__init__()
        inter_channel = 192
        #self.conv1 = conv_relu(in_channel = in_planes, out_channel = 64, kernel = 3, padding = 1)
        #self.conv2 = conv_relu(in_channel = 64, out_channel = inter_channel, kernel = 3, padding = 1)
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = 64, kernel_size = 3,padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = inter_channel, kernel_size = 3, padding = 1)
        self.RFB1 = BasicRFB(inter_channel, inter_channel)
        self.RFB2 = BasicRFB(inter_channel, inter_channel)
        self.RFB3 = BasicRFB(inter_channel, inter_channel)
        # self.RFB4 = BasicRFB(inter_channel, inter_channel)
        # self.RFB5 = BasicRFB(inter_channel, inter_channel)
        # self.RFB6 = BasicRFB(inter_channel, inter_channel)
        self.conv3 = nn.Conv2d(in_channels = inter_channel, out_channels = out_planes, kernel_size = 3, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.relu(self.conv1(x))
        self.conv1_feature = out
        out = self.relu(self.conv2(out))
        self.conv2_feature = out
        out = self.RFB1(out)
        self.RFB1_feature = out
        out = self.RFB2(out)
        self.RFB2_feature = out
        out = self.RFB3(out)
        self.RFB3_feature = out
        # out = self.RFB4(out)
        #out = self.RFB5(out)
        #out = self.RFB6(out)
        out = self.conv3(out)
        return out

class RFNet_v1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(RFNet_v1, self).__init__()
        inter_channel = 64
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = 64, kernel_size = 3, padding = 1)
        # self.conv2 = nn.Conv2d(in_channels = 64, out_channels = inter_channel, kernel_size = 3, padding = 1)
        self.RFB1 = BasicRFB(inter_channel, inter_channel)
        self.RFB2 = BasicRFB(inter_channel, inter_channel)
        self.RFB3 = BasicRFB(inter_channel, inter_channel)
        self.RFB4 = BasicRFB(inter_channel, inter_channel)
        #self.RFB5 = BasicRFB(inter_channel, inter_channel)
        #self.RFB6 = BasicRFB(inter_channel, inter_channel)
        self.conv3 = nn.Conv2d(in_channels = inter_channel, out_channels = out_planes, kernel_size = 3, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.relu(self.conv1(x))
        # out = self.relu(self.conv2(out))
        out = self.RFB1(out)
        out = self.RFB2(out)
        out = self.RFB3(out)
        out = self.RFB4(out)
        #out = self.RFB5(out)
        #out = self.RFB6(out)
        out = self.conv3(out)
        return out

class RFNet_v2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(RFNet_v2, self).__init__()
        inter_channel = 192
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = inter_channel, kernel_size = 3, padding = 1)
        self.RFB1 = BasicRFB(inter_channel, inter_channel)
        self.RFB2 = BasicRFB(inter_channel, inter_channel)
        self.RFB3 = BasicRFB(inter_channel, inter_channel)
        self.RFB4 = BasicRFB(inter_channel, inter_channel)
        self.RFB5 = BasicRFB(inter_channel, inter_channel)
        self.RFB6 = BasicRFB(inter_channel, inter_channel)
        self.conv3 = nn.Conv2d(in_channels = inter_channel, out_channels = out_planes, kernel_size = 3, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.RFB1(out)
        out = self.RFB2(out)
        out = self.RFB3(out)
        out = self.RFB4(out)
        out = self.RFB5(out)
        out = self.RFB6(out)
        out = self.conv3(out)
        return out

# dialted conv with residual learning
class RFNet_rl(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(RFNet_rl, self).__init__()
        inter_channel = 192
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = inter_channel, kernel_size = 3, padding = 1)
        self.RFB1 = BasicRFB(inter_channel, inter_channel)
        self.RFB2 = BasicRFB(inter_channel, inter_channel)
        self.RFB3 = BasicRFB(inter_channel, inter_channel)
        self.conv3 = nn.Conv2d(in_channels = inter_channel, out_channels = out_planes, kernel_size = 3, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        y = self.relu(self.conv1(x))
        self.conv1_feature = y
        y = self.relu(self.conv2(y))
        self.conv2_feature = y
        y = self.RFB1(y)
        self.RFB1_feature = y
        y = self.RFB2(y)
        self.RFB2_feature = y
        y = self.RFB3(y)
        self.RFB3_feature = y
        y = self.conv3(y)
        out = y + x
        return out

class NewRFNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(NewRFNet, self).__init__()
        inter_channel = 192
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = inter_channel, kernel_size = 3, padding = 1)
        self.RFB1 = BasicRFB(inter_channel, inter_channel)
        self.RFB2 = BasicRFB(inter_channel, inter_channel)
        self.RFB3 = BasicRFB(inter_channel, inter_channel)
        #self.RFB4 = BasicRFB(inter_channel, inter_channel)
        #self.RFB5 = BasicRFB(inter_channel, inter_channel)
        #self.RFB6 = BasicRFB(inter_channel, inter_channel)
        self.conv3 = nn.Conv2d(in_channels = inter_channel, out_channels = out_planes, kernel_size = 3, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.RFB1(out)
        out = self.RFB2(out)
        out = self.RFB3(out)
        #out = self.RFB4(out)
        #out = self.RFB5(out)
        #out = self.RFB6(out)
        out = self.conv3(out)
        return out


class VRCNN(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(VRCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes,out_channels=64,kernel_size=3,padding=1)
        # self.conv1 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, padding=2)
        self.MyInc1 = MyInception(in_channel=64, out1=16, out2=32, kernel1=5, kernel2=3, padding1=2, padding2=1)
        self.MyInc2 = MyInception(in_channel=48, out1=16, out2=32, kernel1=3, kernel2=1, padding1=1, padding2=0)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=out_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):

        out = self.relu(self.conv(x))
        # out = self.relu(self.conv1(out))
        # self.conv1_feature = out
        out = self.relu(self.MyInc1(out))
        self.MyInc1_feature = out
        out = self.relu(self.MyInc2(out))
        self.MyInc2_feature = out
        out = self.conv2(out)
        self.conv2_feature = out
        return out

# VRCNN with residual learning
class VRCNN_rl(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(VRCNN_rl, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=64, kernel_size=5, padding=2)
        self.MyInc1 = MyInception(in_channel=64, out1=16, out2=32, kernel1=5, kernel2=3, padding1=2, padding2=1)
        self.MyInc2 = MyInception(in_channel=48, out1=16, out2=32, kernel1=3, kernel2=1, padding1=1, padding2=0)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=out_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        y = self.relu(self.conv1(x))
        self.conv1_feature = y
        y = self.relu(self.MyInc1(y))
        self.MyInc1_feature = y
        y = self.relu(self.MyInc2(y))
        self.MyInc2_feature = y
        y = self.conv2(y)
        self.fea = y
        out = x + y
        return out

class VRCNN_H(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(VRCNN_H, self).__init__()
        self.conv1 = conv_relu(in_channel=in_planes, out_channel=32, kernel=5, padding=2)
        self.MyInc1 = MyInception(in_channel=32, out1=8, out2=16, kernel1=5, kernel2=3, padding1=2, padding2=1)
        self.MyInc2 = MyInception(in_channel=24, out1=8, out2=16, kernel1=3, kernel2=1, padding1=1, padding2=0)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=out_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.MyInc1(out))
        out = self.relu(self.MyInc2(out))
        out = self.conv2(out)
        return out

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        # 加了padding sub_image大小在网络中不变
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)  # padding = 4
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)  # padding = 2
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2) # padding = 2
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        self.conv1_feature = x
        x = self.relu(self.conv2(x))
        self.conv2_feature = x
        x = self.conv3(x)
        return x

# SRCNN with residual learning
class SRCNN_rl(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN_rl, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)  # padding = 4
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)  # padding = 2
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2) # padding = 2
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        self.conv1_feature = y
        y = self.relu(self.conv2(y))
        self.conv2_feature = y
        y = self.conv3(y)
        out = y + x
        return out

class Dianet(nn.Module):
    def __init__(self, in_planes = 1, out_planes = 1):
        super(Dianet, self).__init__()
        inter_channel = 192
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=inter_channel, kernel_size=3, padding=1)
        self.RFB_De1 = BasicRFB_De(inter_channel, inter_channel)
        self.RFB_De2 = BasicRFB_De(inter_channel, inter_channel)
        self.RFB_De3 = BasicRFB_De(inter_channel, inter_channel)
        self.conv3 = nn.Conv2d(in_channels=inter_channel, out_channels=out_planes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.RFB_De1(out)
        out = self.RFB_De2(out)
        out = self.RFB_De3(out)
        out = self.conv3(out)
        return out

class BasicRFB_De(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1,map_reduce=6):
        super(BasicRFB_De, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce  # inter_planes = 192 // 6 = 32

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride, bn=False),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=1, relu=False, bn=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=False),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=3, stride=stride, padding=1, bn=False),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False, bn=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=False),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1, bn=False),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1, bn=False),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False, bn=False)
                )
        '''
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,5), stride=1, padding=(0,2)),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=(5,1), stride=stride, padding=(2,0)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )
        '''
        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False, bn=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False, bn=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        #x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


class DeformConvNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DeformConvNet, self).__init__()
        inter_channel = 192
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = 64, kernel_size = 3,padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.deform1 = DeformConv2d(inc = 128, outc = 192, kernel_size = 3, padding=1, bias=False, modulation=False)
        self.deform2 = DeformConv2d(inc = 192, outc = 192, kernel_size = 3, padding=1, bias=False, modulation=False)
        # self.RFB1 = BasicRFB(inter_channel, inter_channel)
        # self.RFB2 = BasicRFB(inter_channel, inter_channel)
        # self.RFB3 = BasicRFB(inter_channel, inter_channel)
        # self.RFB4 = BasicRFB(inter_channel, inter_channel)
        # self.RFB5 = BasicRFB(inter_channel, inter_channel)
        # self.RFB6 = BasicRFB(inter_channel, inter_channel)
        self.conv3 = nn.Conv2d(in_channels = 192, out_channels = out_planes, kernel_size = 3, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.deform1(out))
        out = self.relu(self.deform2(out))

        # out = self.RFB1(out)
        # out = self.RFB2(out)
        # out = self.RFB3(out)

        # out = self.RFB4(out)
        #out = self.RFB5(out)
        #out = self.RFB6(out)
        out = self.conv3(out)
        return out


class DeformConvNet_v2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DeformConvNet_v2, self).__init__()
        inter_channel = 192
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = 64, kernel_size = 3,padding = 1)
        # self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.deform1 = DeformConv2d(inc = 64, outc = 128, kernel_size = 3, padding=1, bias=False, modulation=False)
        self.deform2 = DeformConv2d(inc = 128, outc = 128, kernel_size = 3, padding=1, bias=False, modulation=False)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = out_planes, kernel_size = 3, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.relu(self.conv1(x))
        # out = self.relu(self.conv2(out))
        out = self.relu(self.deform1(out))
        out = self.relu(self.deform2(out))

        out = self.conv3(out)
        return out