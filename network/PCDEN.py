import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules import *
from .init_weights import init_weights
from .ResNet import resnet50, resnet101, resnet152
from .MobileNetV2 import mobilenet_v2
from .vgg import VGG


class PCDEN(nn.Module):
    #  position capturing detail enhancing network
    # backbone:  resnet50, VGG, mobilenet_v2
    def __init__(self, backbone_name='resnet50', mid_ch=128, bottleneck_num=2, **kwargs):
        super(PCDEN, self).__init__()      

        self.backbone_name=backbone_name
        eout_channels=[64, 256, 512, 1024, 2048]
        if backbone_name.find('VGG')!=-1:
            self.backbone  = VGG('rgb')
            eout_channels=[64, 128, 256, 512, 512]
        elif backbone_name=='mobilenet_v2':
            self.backbone  = mobilenet_v2(pretrained=False)
            eout_channels=[16, 24, 32, 96, 320]
        else:# default: Resnet
            self.backbone  = eval(backbone_name)(pretrained=False)

        out_ch=1
      
        # Encoder
        self.eside1=ConvModule(eout_channels[0], mid_ch)
        self.eside2=ConvModule(eout_channels[1], mid_ch)
        self.eside3=ConvModule(eout_channels[2], mid_ch)
        self.eside4=ConvModule(eout_channels[3], mid_ch)
        self.eside5=ConvModule(eout_channels[4], mid_ch)

        # Decoder
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.spcm1=SPCM1(mid_ch)
        self.spcm2=SPCM(mid_ch)
        self.spcm3=SPCM(mid_ch)
        self.spcm4=SPCM(mid_ch)
        self.spcm5=SPCM5(mid_ch)

        
        self.sdem1=SDEM(mid_ch)
        self.sdem2=SDEM(mid_ch)
        self.sdem3=SDEM(mid_ch)
        self.sdem4=SDEM(mid_ch)
        self.sdem5=SDEM(mid_ch)
        

        self.dec1=AFFD(c1=mid_ch, c2=mid_ch, val_num=2, n=bottleneck_num)
        self.dec2=AFFD(c1=mid_ch, c2=mid_ch, val_num=2, n=bottleneck_num)
        self.dec3=AFFD(c1=mid_ch, c2=mid_ch, val_num=2, n=bottleneck_num)
        self.dec4=AFFD(c1=mid_ch, c2=mid_ch, val_num=2, n=bottleneck_num)
        self.dec5=AFFD(c1=mid_ch, c2=mid_ch, val_num=1, n=bottleneck_num)

        self.dside1 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside2 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside3 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside4 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside5 = nn.Conv2d(mid_ch,out_ch,3,padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # encoder
        if self.backbone_name!='VGG':
            outs = self.backbone(inputs)
            c1, c2, c3, c4, c5 = outs
        else:
            x1_rgb = self.backbone.conv1(inputs)
            x2_rgb = self.backbone.conv2(x1_rgb)
            x3_rgb = self.backbone.conv3(x2_rgb)
            x4_rgb = self.backbone.conv4(x3_rgb)
            x5_rgb = self.backbone.conv5(x4_rgb)

            c1=x1_rgb
            c2=x2_rgb
            c3=x3_rgb
            c4=x4_rgb
            c5=x5_rgb

        c1=self.eside1(c1)
        c2=self.eside2(c2)
        c3=self.eside3(c3)
        c4=self.eside4(c4)
        c5=self.eside5(c5)

        # SPCM
        ca1=self.spcm1(c1,c2)
        ca2=self.spcm2(c1,c2,c3)
        ca3=self.spcm3(c2,c3,c4)
        ca4=self.spcm4(c3,c4,c5)
        ca5=self.spcm5(c4,c5)

        # SDEM
        ca1=self.sdem1(ca1)
        ca2=self.sdem2(ca2)
        ca3=self.sdem3(ca3)
        ca4=self.sdem4(ca4)
        ca5=self.sdem5(ca5)

        # AFFD
        up5=self.dec5(ca5)
        up4=self.dec4(ca4, self.upsample2(up5))
        up3=self.dec3(ca3, self.upsample2(up4))
        up2=self.dec2(ca2, self.upsample2(up3))
        up1=self.dec1(ca1, self.upsample2(up2))

        d1=self.dside1(up1)
        d2=self.dside2(up2)
        d3=self.dside3(up3)
        d4=self.dside4(up4)
        d5=self.dside5(up5)

        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)
        S5 = F.interpolate(d5, size=(H, W), mode='bilinear', align_corners=True)

        return S1, S2, S3, S4, S5, torch.sigmoid(S1),torch.sigmoid(S2),torch.sigmoid(S3),torch.sigmoid(S4),torch.sigmoid(S5)
