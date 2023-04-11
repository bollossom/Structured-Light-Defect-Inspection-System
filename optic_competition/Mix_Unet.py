import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

from fastai.layers import PixelShuffle_ICNR, ConvLayer, SelfAttention


class Embedding3D(nn.Module):
    def __init__(self, corpus_size, output_shape):
        super().__init__()
        self.embedding = nn.parameter.Parameter(torch.randn(corpus_size, *output_shape))

    def forward(self, inputs):
        return torch.stack([self.embedding[o] for o in inputs], dim=0)


# -------------ASPP----------------------------------------
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
                     [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                         nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                                         nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False),
                                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# -------------ASPP----------------------------------------


# -------------FPN----------------------------------------
class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch * 2),
                           nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1))
             for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs: list, last_layer):
        hcs = [F.interpolate(c(x), scale_factor=2 ** (len(self.convs) - i), mode='bilinear')
               for i, (c, x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


# -------------FPN----------------------------------------


# -------------Attention----------------------------------------
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


# -------------Attention----------------------------------------


# -------------decoder----------------------------------------
class UnetBlock(nn.Module):
    def __init__(self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False,
                 self_attention: bool = False, **kwargs):
        super().__init__()
        # self.shuf=nn.ConvTranspose2d(up_in_c, up_in_c // 2,kernel_size=2,
        #                               stride=2,padding=0,bias=True)
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None, xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in, left_in):
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


# -------------decoder----------------------------------------


# ------------ResNet50---------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPANSION = 4


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, EXPANSION * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(EXPANSION * planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != EXPANSION * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, EXPANSION * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(EXPANSION * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        woha = self.shortcut(x)
        out += woha
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.in_planes = 64

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)  # 3 times

        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)  # 4 times

        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)  # 6 times

        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)  # 3 times



    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = EXPANSION * planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 3, stride=2)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

# ------------ResNet50---------------------------------------
# --------------------------------SpanConv Block -----------------------------------#
class SpanConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpanConv, self).__init__()
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size

        self.point_wise_1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        self.depth_wise_1 = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size - 1) // 2,
                                    groups=out_channels,
                                    bias=True)

        self.point_wise_2 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        self.depth_wise_2 = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size - 1) // 2,
                                    groups=out_channels,
                                    bias=True)


    def forward(self, x):  #
        out_tmp_1 = self.point_wise_1(x)  #
        out_tmp_1 = self.depth_wise_1(out_tmp_1)  #

        out_tmp_2 = self.point_wise_2(x)  #
        out_tmp_2 = self.depth_wise_2(out_tmp_2)  #

        out = out_tmp_1 + out_tmp_2

        return out
# --------------------------------SpanConv Block -----------------------------------#


# -------------Mix_unet----------------------------------------
class Mix_Unet(nn.Module):
    def __init__(self, stride=1, **kwargs):
        super().__init__()
        # encoder

        encoder_block = ResNet([2, 3, 4, 2])

        self.enc0 = nn.Sequential(encoder_block.conv1, encoder_block.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1), encoder_block.layer1)
        self.enc2 = encoder_block.layer2
        self.enc3 = encoder_block.layer3
        self.enc4 = encoder_block.layer4
        # aspp with customized dilatations
        self.aspp = ASPP(2048, 256, out_c=512, dilations=[stride * 1, stride * 2, stride * 3, stride * 4])

        # decoder
        self.dec4 = UnetBlock(512, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([512, 256, 128, 64], [16] * 4)
        self.deconv=nn.ConvTranspose2d(32+16*4,64,kernel_size=2,stride=2,padding=0,bias=True)
        self.final_block =nn.Sequential(nn.ConvTranspose2d(32+16*4,48,kernel_size=2,stride=2,padding=0,bias=True),
                                        nn.BatchNorm2d(48),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=48,out_channels=1,kernel_size=1),
                                        )


    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)

        dec3 = self.dec4(enc5, enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x=self.final_block(x)
        return x


# -------------Mix_unet----------------------------------------


# -------------decoder----------------------------------------


# model=Mix_Unet()
# x=torch.zeros(2,1,480,640)
# print(model(x).shape)
# from torchsummary import summary
# ch = 1
# h = 256
# w = 256
# summary(model, input_size=(ch, h, w), device='cpu')

