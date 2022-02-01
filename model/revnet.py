import torch
import torch.nn as nn

from model import nn as onn
from model import utils as MU
from model.reversible import ReversibleLayers


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            planes: int,
            stride: int = 1,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            planes: int,
            stride: int = 1,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)

        return out


class RevNet(nn.Module):
    def __init__(self, block, num_classes, num_block, k=1,
                 block_feature=(64, 128, 256, 512),
                 groups=1,
                 width_per_group=64,
                 norm_layer=None,
                 parameter_share=1,
                 **kwargs
                 ):
        super(RevNet, self).__init__()
        self.inplanes = 32
        self.expansion = block.expansion
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = []
        for idx, (feature, num_b) in enumerate(zip(block_feature, num_block)):
            feature *= self.expansion
            if (idx != 0): layers.append([onn.Upsample(scale_factor=0.5), onn.Upsample(scale_factor=0.5)])
            layers.append([onn.ChannelChanger(feature // k), onn.ChannelChanger(feature // k)])
            layers.append([
                block(feature // k, groups=groups, base_width=width_per_group, norm_layer=norm_layer),
                block(feature // k, groups=groups, base_width=width_per_group, norm_layer=norm_layer)])

            for b_idx in range(1, num_b):
                # if (b_idx == num_b // 2):
                #     layers.append([onn.Upsample(scale_factor=0.5), onn.Upsample(scale_factor=0.5)])
                if (b_idx % parameter_share != 0):
                    layers.append(layers[-1])
                else:
                    layers.append([
                        block(feature // k, groups=groups, base_width=width_per_group, norm_layer=norm_layer),
                        block(feature // k, groups=groups, base_width=width_per_group, norm_layer=norm_layer)])

        self.layers = ReversibleLayers(MU.deeplist2module(layers), innerlayers=[BasicBlock, Bottleneck])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block_feature[-1] * block.expansion * (3 - k), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # TODO fix double input
        x = self.layers(torch.cat([x, x], dim=1))

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def revnet18(num_classes, **kwargs):
    return RevNet(block=BasicBlock, num_classes=num_classes, num_block=[1, 1, 1, 1], **kwargs)


def revnet18_r2(num_classes, **kwargs):
    return RevNet(block=BasicBlock, num_classes=num_classes, num_block=[1, 1, 1, 1], block_feature=(90, 180, 360, 720),
                  **kwargs)


def revnet34(num_classes, **kwargs):
    return RevNet(block=BasicBlock, num_classes=num_classes, num_block=[2, 2, 3, 2], **kwargs)


def revnet50(num_classes, **kwargs):
    return RevNet(block=Bottleneck, num_classes=num_classes, num_block=[1, 2, 3, 2], **kwargs)


def revnet104(num_classes, **kwargs):
    return RevNet(block=Bottleneck, num_classes=num_classes, num_block=[2, 2, 11, 2], **kwargs)


def revnet152(num_classes, **kwargs):
    return RevNet(block=Bottleneck, num_classes=num_classes, num_block=[1, 4, 18, 2], **kwargs)


def revnext50_32x4d(num_classes, **kwargs):
    return RevNet(block=Bottleneck, num_classes=num_classes, num_block=[1, 2, 3, 2], groups=32, width_per_group=4,
                  **kwargs)


def revnext104_32x8d(num_classes, **kwargs):
    return RevNet(block=Bottleneck, num_classes=num_classes, num_block=[1, 2, 12, 2], groups=32, width_per_group=4,
                  **kwargs)


def wide_revnet50_2(num_classes, **kwargs):
    return RevNet(block=Bottleneck, num_classes=num_classes, num_block=[1, 2, 3, 2], width_per_group=64 * 2, **kwargs)


def wide_revnet104_2(num_classes, **kwargs):
    return RevNet(block=Bottleneck, num_classes=num_classes, num_block=[1, 2, 12, 2], width_per_group=64 * 2, **kwargs)


models = {"revnet18": revnet18, "revnet18_r2": revnet18_r2, "revnet34": revnet34, "revnet50": revnet50,
          "revnet101": revnet104, "revnet152": revnet152, "wide_revnet50_2": wide_revnet50_2,
          "wide_revnet101_2": wide_revnet104_2, "revnext50_32x4d": revnext50_32x4d,
          "revnext101_32x8d": revnext104_32x8d}
