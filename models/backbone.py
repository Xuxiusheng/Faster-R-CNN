import math
import os
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------#
        #   假设输入进来的图片是600,600,3
        # -----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(19)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        feature_2 = x["feature_2"] + F.interpolate(self.conv1(x["feature_3"]), x["feature_2"].shape[-2:], mode="bilinear", align_corners=True)
        x["feature_2"] = feature_2
        feature_1 = x["feature_1"] + F.interpolate(self.conv2(x["feature_2"]), x["feature_1"].shape[-2:], mode="bilinear", align_corners=True)
        x["feature_1"] = feature_1

        return x


def resnet50(pretrained=True):
    pretrain_model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        if not os.path.exists("./setting/resnet50-19c8e357.pth"):
            print("pretrain backbone file not found, start downloading....")
            state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth",
                                                model_dir="./setting")
        else:
            print("pretrain backbone file found, start loading...")
            state_dict = torch.load("./setting/resnet50-19c8e357.pth")
        pretrain_model.load_state_dict(state_dict)
        print("pretrain backbone has been loaded!")

    extractor = list([pretrain_model.conv1, pretrain_model.bn1, pretrain_model.relu,
                      pretrain_model.maxpool, pretrain_model.layer1, pretrain_model.layer2,
                      pretrain_model.layer3])

    extractor = nn.Sequential(*extractor)

    classifier = list([pretrain_model.layer4, pretrain_model.avgpool])
    classifier = nn.Sequential(*classifier)

    return extractor, classifier

if __name__ == "__main__":
    resnet = ResNet(Bottleneck, [3, 4, 6, 3])
    # print(resnet.device)
    # summary(resnet, (3, 600, 600), device="cpu")
    resnet(torch.rand((4, 3, 600, 600)))

