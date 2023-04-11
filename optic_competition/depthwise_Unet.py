import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
class Depthwiseconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Depthwiseconv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
class normal_conv_layer_BN_Relu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(normal_conv_layer_BN_Relu, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=strides,padding=padding)
        self.bn1=nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.bn1(x1)
        out=F.relu(x2)
        return  out
class depthwise_separable_residual_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(depthwise_separable_residual_conv, self).__init__()
        self.conv1_1_1=nn.Conv2d(in_channels,2*in_channels,kernel_size=1)
        self.bn1=nn.BatchNorm2d(2*in_channels)
        self.depth_conv=Depthwiseconv(2*in_channels,2*in_channels)
        self.bn2=nn.BatchNorm2d(2*in_channels)
        self.conv1_1_2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
    def forward(self,x):
        x1=self.conv1_1_1(x)
        x2=self.bn1(x1)
        x3=F.relu(x2)
        x4 = self.depth_conv(x3)
        x5 = self.bn2(x4)
        x6 = F.relu(x5)
        x7=self.conv1_1_2(x6)
        out=x7+x
        return  out
class transposed_conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(transposed_conv, self).__init__()

        self.conv1=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,
                                      stride=strides,padding=0,bias=True)
    def forward(self, x):
        out=self.conv1(x)
        return out
class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_block, self).__init__()
        self.conv1=normal_conv_layer_BN_Relu(in_channels, out_channels)
        self.conv2=depthwise_separable_residual_conv(out_channels,out_channels)
    def forward(self, x):
        x1=self.conv1(x)
        x2=self.conv2(x1)
        out=x2
        return out
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()
        self.conv1=normal_conv_layer_BN_Relu(in_channels, out_channels)
        self.conv2=depthwise_separable_residual_conv(out_channels,out_channels)
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        out=x
        return out
class IUnet(nn.Module):
    def __init__(self):
        super(IUnet, self).__init__()
        self.encoder_block1=encoder_block(1,32)
        self.encoder_block2=encoder_block(32,64)
        self.encoder_block3 = encoder_block(64, 128)
        self.encoder_block4 = encoder_block(128, 256)
        self.encoder_block5=encoder_block(256,512)

        self.deconv1=transposed_conv(512,256)
        self.decoder_block1=decoder_block(512,256)
        self.deconv2 = transposed_conv(256, 128)
        self.decoder_block2=decoder_block(256,128)
        self.deconv3 = transposed_conv(128, 64)
        self.decoder_block3 = decoder_block(128,64)
        self.deconv4 = transposed_conv(64, 32)
        self.decoder_block4 = decoder_block(64,32)
        self.end = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
    def forward(self, x):
        #1*224*224
        conv1=self.encoder_block1(x)#输出为32*224*224
        pool1=F.max_pool2d(conv1,2)#输出为32*112*112
        # print('pool1', pool1.shape)
        conv2=self.encoder_block2(pool1)#输出为64*112*112
        pool2=F.max_pool2d(conv2,2)#输出为64*56*56
        # print('pool2', pool2.shape)
        conv3=self.encoder_block3(pool2)#128*56*56
        # print('conv3',conv3.shape)
        pool3=F.max_pool2d(conv3,2)#128*28*28
        # print('pool3',pool3.shape)
        conv4=self.encoder_block4(pool3)#256*28*28
        pool4=F.max_pool2d(conv4,2)#256*14*14

        conv5=self.encoder_block5(pool4)#512*14*14

        convt1=self.deconv1(conv5)#256*28*28
        concat1=torch.cat([convt1,conv4],dim=1)#512*28*28

        conv6=self.decoder_block1(concat1)#256*28*28

        convt2=self.deconv2(conv6)#128*56*56
        concat2=torch.cat([convt2,conv3],dim=1)#256*56*56

        conv7=self.decoder_block2(concat2)#128*112*112

        convt3=self.deconv3(conv7)#64*112*112
        concat3=torch.cat([convt3,conv2],dim=1)#128*112*112

        conv8=self.decoder_block3(concat3)#64*112*112

        convt4 = self.deconv4(conv8)  # 32*224*224
        concat4 = torch.cat([convt4, conv1], dim=1)  # 64*224*224

        end=self.decoder_block4(concat4)
        # print(end.shape)
        out=self.end(end)
        return out
# model=IUnet()
# ch = 1
# h = 480
# w = 640
# summary(model, input_size=(ch, h, w), device='cpu')
#7373,345
# x=torch.randn(1,1,224,224)
# print(model(x).shape)