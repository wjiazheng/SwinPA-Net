import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.swin import swin_B, swin_L

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class SwinPA(nn.Module):
    def __init__(self, out_planes=1, encoder='swin_B'):
        super(SwinPA, self).__init__()
        self.encoder = encoder
        if self.encoder == 'swin_L':
            mutil_channel = [192, 384, 768, 1536]
            self.backbone = swin_L()
        elif self.encoder == 'swin_B':
            mutil_channel = [128, 256, 512, 1024]
            self.backbone = swin_B()
        elif self.encoder == 'pvt':
            mutil_channel = [64, 128, 320, 512]
            self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
            path = './pretrained_pth/pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
       
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dmc1 = DMC1([mutil_channel[2], mutil_channel[3]], width=mutil_channel[2], up_kwargs=up_kwargs)
        self.dmc2 = DMC2([mutil_channel[1], mutil_channel[2], mutil_channel[3]], width=mutil_channel[1],
                          up_kwargs=up_kwargs)
        self.dmc3 = DMC3([mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3]],
                          width=mutil_channel[0], up_kwargs=up_kwargs)
        self.DMC_fusion = DMC_fusion(mutil_channel, up_kwargs=up_kwargs)
        self.lpa1 = LPA(in_channel=mutil_channel[0])
        self.lpa2 = LPA(in_channel=mutil_channel[1])
        self.lpa3 = LPA(in_channel=mutil_channel[2])
        self.lpa4 = LPA(in_channel=mutil_channel[3])

        self.decoder4 = BasicConv2d(mutil_channel[3], mutil_channel[2], 3, padding=1)
        self.decoder3 = BasicConv2d(mutil_channel[2], mutil_channel[1], 3, padding=1)
        self.decoder2 = BasicConv2d(mutil_channel[1], mutil_channel[0], 3, padding=1)
        self.decoder1 = nn.Sequential(nn.Conv2d(mutil_channel[0], 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(64, out_planes, kernel_size=1, stride=1))

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        # x1 = self.dmc3(x1, x2, x3, x4)  #乘法融合
        # x2 = self.dmc2(x2, x3, x4)
        # x3 = self.dmc1(x3, x4)
        # x4 = x4
        x1, x2, x3, x4 = self.DMC_fusion(x1, x2, x3, x4) #整理为一个模块

        y1 = self.lpa1(x1)
        y2 = self.lpa2(x2)
        y3 = self.lpa3(x3)
        y4 = self.lpa4(x4)

        d4 = self.decoder4(torch.cat((self.decoder4(self.upsample(y4)), y3), dim=1))
        d3 = self.decoder3(torch.cat((self.decoder3(self.upsample(d4)), y2), dim=1))
        d2 = self.decoder2(torch.cat((self.decoder2(self.upsample(d3)), y1), dim=1))
        d1 = self.decoder1(d2)
        d1 = F.interpolate(d1, scale_factor=4, mode='bilinear')

        return d1


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DMC1(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None):
        super(DMC1, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, padding=0, bias=False),  # 调整通道数1*1卷积
            nn.BatchNorm2d(width))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feat = feats[0] * feats[1]
        return feat


class DMC2(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None):
        super(DMC2, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)

        feat = feats[0] * feats[1] * feats[2]
        return feat


class DMC3(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None):
        super(DMC3, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        feat = feats[0] * feats[1] * feats[2] * feats[3]
        return feat


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class LPA(nn.Module):
    def __init__(self, in_channel):
        super(LPA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)
        x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]

        x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        x4 = self.ca(x) * x
        x4 = self.sa(x4) * x4
        x = x3 + x4
        return x


class DMC_fusion(nn.Module):
    def __init__(self, in_channels, up_kwargs=None):
        super(DMC_fusion, self).__init__()

        self.up_kwargs = up_kwargs

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels[3], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3, x4):
        x4_1 = x4
        x4_2 = F.interpolate(self.conv4_2(x4_1), scale_factor=2, **self.up_kwargs)
        x3_1 = x4_2 * (self.conv3_1(x3))
        x3_2 = F.interpolate(self.conv3_2(x3_1), scale_factor=2, **self.up_kwargs)
        x2_1 = x3_2 * (self.conv2_1(x2))
        x2_2 = F.interpolate(self.conv2_2(x2_1), scale_factor=2, **self.up_kwargs)
        x1_1 = x2_2 * (self.conv1_1(x1))

        return x1_1, x2_1, x3_1, x4_1