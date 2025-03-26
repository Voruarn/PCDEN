import torch
import torch.nn as nn
import torch.nn.functional as F
from network.init_weights import init_weights
from timm.models.layers import DropPath, trunc_normal_


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    # default_act = nn.SiLU()  # default activation
    default_act=nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ChannelAttention(nn.Module):
    # Channel-attention module
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))

######

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv2Module(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(Conv2Module, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size), nn.ReLU(inplace=True),)
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class SDEM(nn.Module):
    # Salient Detail Enhancing Module
    def __init__(self, in_channel):
        super(SDEM, self).__init__()
        self.cbam=CBAM(in_channel)
        self.in_ch=in_channel
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel*2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x=self.cbam(x)
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :self.in_ch, :, :], out2[:, self.in_ch:, :, :]

        return F.relu(w * out1 + b, inplace=True)


class SPCM(nn.Module):
    # SPCM: spatial position capturing module
    def __init__(self, cur_channel):
        super(SPCM, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        
        self.fuse = ConvModule(cur_channel*2, cur_channel)
        self.sigmoid = nn.Sigmoid()
                                       

    def forward(self, x_pre, x_cur, x_lat):
        pre_ds=self.downsample2(x_pre)
        lat_up=self.upsample2(x_lat)
        
        x=torch.cat([pre_ds+x_cur, lat_up+x_cur], dim=1)
        x= self.sigmoid(self.fuse(x))
        x=x*x_cur+x_cur
        return x


class SPCM1(nn.Module):
    def __init__(self, cur_channel):
        super(SPCM1, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.fuse = ConvModule(cur_channel, cur_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cur, x_lat):
        lat_up=self.upsample2(x_lat)

        x= self.sigmoid(self.fuse(lat_up+x_cur))
        x=x*x_cur+x_cur
        return x


class SPCM5(nn.Module):
    def __init__(self, cur_channel):
        super(SPCM5, self).__init__()
        self.downsample2 = nn.MaxPool2d(2, stride=2)

        self.fuse = ConvModule(cur_channel, cur_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_pre, x_cur):
        pre_ds=self.downsample2(x_pre)
       
        x= self.sigmoid(self.fuse(pre_ds+x_cur))
        x=x*x_cur+x_cur
        return x

class AFFD(nn.Module):
    # AFFD: Adaptive Feature Fusion Decoder
    def __init__(self, c1, c2, val_num=2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
  
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) if val_num==2 else nn.Identity()

    def forward(self, x1, x2=None):
        if x2==None:
            x=x1
        else:
            nw = F.softmax(self.weight, dim=0)
            x=nw[0] * x1 + nw[1] * x2

        a, b = self.cv1(x).split((self.c, self.c), 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


if __name__=='__main__':
    print('Hello')