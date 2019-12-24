# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import squeezenet_mod
#early stopping 
from torchsample.callbacks import EarlyStopping

 

cfg = {'PicaNet': "GGLLL",
       'Size': [28, 28, 28, 56, 112, 224],
       'Channel': [1024, 512, 512, 256, 128, 64],
       'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}


class Unet(nn.Module):
    def __init__(self, cfg={'PicaNet': "GGLLL",
       'Size': [28, 28, 28, 56, 112, 224],
       'Channel': [1024, 512, 512, 256, 128, 64],
       'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}):
        super(Unet, self).__init__()
        self.encoder = Encoder()
        self.decoder = nn.ModuleList()
        self.cfg = cfg
        for i in range(5):
            assert cfg['PicaNet'][i] == 'G' or cfg['PicaNet'][i] == 'L'
            self.decoder.append(
                DecoderCell(size=cfg['Size'][i],
                            in_channel=cfg['Channel'][i],
                            out_channel=cfg['Channel'][i + 1],
                            mode=cfg['PicaNet'][i]))
        self.decoder.append(DecoderCell(size=cfg['Size'][5],
                                        in_channel=cfg['Channel'][5],
                                        out_channel=1,
                                        mode='C'))

    def forward(self, *input):
        if len(input) == 2:
            x = input[0]
            tar = input[1]
            test_mode = False
        if len(input) == 3:
            x = input[0]
            tar = input[1]
            test_mode = input[2]
        if len(input) == 1:
            x = input[0]
            tar = None
            test_mode = True
        en_out = self.encoder(x)
        # print(len(en_out))
        dec = None
        pred = []
        for i in range(6):
            # print(en_out[5 - i].size())
            dec, _pred = self.decoder[i](en_out[5 - i], dec)
            pred.append(_pred)
        loss = 0
        if not test_mode:
            for i in range(6):
                loss += self.cfg['loss_ratio'][5 - i]*F.mse_loss(pred[5 - i], tar)
		
                # print(float(loss))

                if tar.size()[2] > 28:
                    tar = F.max_pool2d(tar, 2, 2)
        return pred, loss

########### Added Fire Module #######

class Fire(torch.nn.Module):
    def __init__(self,inchn,sqzout_chn,exp1x1out_chn,exp3x3out_chn):
        super(Fire,self).__init__()
        self.inchn = inchn
        self.squeeze = torch.nn.Conv2d(inchn,sqzout_chn,kernel_size=1)
        self.squeeze_act = torch.nn.ReLU(inplace=True)
        self.expand1x1 = torch.nn.Conv2d(sqzout_chn,exp1x1out_chn,kernel_size=1)
        self.expand1x1_act = torch.nn.ReLU(inplace=True)
        self.expand3x3 = torch.nn.Conv2d(sqzout_chn,exp3x3out_chn,kernel_size=3, padding=1)
        self.expand3x3_act = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.squeeze_act(self.squeeze(x))
        return torch.cat([
                self.expand1x1_act(self.expand1x1(x)),
                self.expand3x3_act(self.expand3x3(x))
                ], 1)

######## Added squeeze layers ##########

def squeeze_layers():
    return torch.nn.Sequential(
    # torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
    torch.nn.Conv2d(3, 96, kernel_size=7, stride=2),
    torch.nn.ReLU(inplace=True),
    torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
    Fire(96, 16, 64, 64),
    Fire(128, 16, 64, 64),
    # torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
    Fire(128, 32, 128, 128),
    torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
    Fire(256, 32, 128, 128),
    Fire(256, 48, 192, 192),
    Fire(384, 48, 192, 192),
    Fire(384, 64, 256, 256),
    torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
    Fire(512, 64, 256, 256),
    # torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
    # Fire(512, 64, 256, 256),
)
#     return torch.nn.Sequential(
#             torch.nn.Conv2d(3,64,kernel_size=7,stride=2),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=False),
#             Fire(64,16,64,64),
#             Fire(128,16,64,64),
#             torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=False),
#             Fire(128,32,128,128),
#             Fire(256,32,128,128),
#             torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=False),
#             Fire(256,48,192,192),
#             Fire(384,48,192,192),
#             Fire(384,64,256,256),
#             Fire(512,64,256,256),
#     )

# def make_layers(cfg, in_channels):
#     layers = []
#     dilation_flag = False
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         elif v == 'm':
#             layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
#             dilation_flag = True
#         elif v=='K':
#             layers += [nn.MaxPool2d(kernel_size=7, stride=2)]
#         elif v=='S':
#             layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
#         elif v=='N':
#             layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
#         elif v=='V':
#             layers += [nn.MaxPool2d(kernel_size=13, stride=1)]
#         else:
#             if not dilation_flag:
#                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3,  padding=1)
#             else:
#                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
#             layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     print(layers)
#     # layers = list(resnet.children())[:-1]
#     return nn.Sequential(*layers[:-2])


# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # configure = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'm', 512, 512, 512, 'm']
        # configure = [96,96,'K',128,128,256,'S',256,384,384,512,512,'S',512,'m']
        configure=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.seq = squeeze_layers() ######## Added this line ############
        # print("Printing from Encoder")
        # print(self.seq)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12)  # fc6 in paper
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1, 1)  # fc7 in paper

    def forward(self, *input):
        x = input[0]
        # print(len(self.seq(x)))
        # conv1=self.seq[:2](x)
        conv1 = F.adaptive_avg_pool2d(self.seq[:2](x),[224,224]).squeeze()
        conv2=F.adaptive_avg_pool2d(self.seq[2:6](conv1),[112,112]).squeeze()
        # conv3 = self.seq[6:8](conv2)
        conv3=F.adaptive_avg_pool2d(self.seq[6:8](conv2),[56,56]).squeeze()
        # conv4 = self.seq[8:11](conv3)
        conv4=F.adaptive_avg_pool2d(self.seq[8:11](conv3),[28,28]).squeeze()
        # conv5= self.seq[11:](conv4)
        conv5 = F.adaptive_avg_pool2d(self.seq[11:](conv4), [28, 28]).squeeze()
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        # conv3 = self.seq[6](conv2)
        # conv4 = self.seq[7](conv3)
        # conv5 = self.seq[8](conv4)
        # conv2=torch.reshape(conv1,(512*5,26,26))
        # conv3= F.adaptive_avg_pool2d(conv1,[28,28]).squeeze()
        # print(conv3.size())
        # conv1 = self.seq[:4](x)
        # conv2 = self.seq[4:9](conv1)
        # conv3 = self.seq[9:16](conv2)
        # conv4 = self.seq[16:23](conv3)
        # conv5 = self.seq[23:](conv4)
        # conv6 = self.conv6(conv5)
        # # conv5=conv3
        # conv7 = self.conv7(conv6)
        # conv2 = nn.Conv2d(128, 256,3)(conv2)
        conv2= F.adaptive_avg_pool2d(conv2.reshape(5,128,224,112),[112,112]).squeeze()
        conv1 = F.adaptive_avg_pool2d(conv2.reshape(5, 64, 112, 224), [224, 224]).squeeze()
        # print("conv1", conv1.size())
        # print("conv2", conv2.size())
        # print("conv3", conv3.size())
        # print("conv4", conv4.size())
        # print("conv5", conv5.size())
        # print("conv6", conv6.size())
        # print("conv7", conv7.size())

        return conv1, conv2, conv3, conv4, conv5, conv7

        # return conv1, conv2, conv3, conv4, conv5, conv7


class DecoderCell(nn.Module):
    def __init__(self, size, in_channel, out_channel, mode):
        super(DecoderCell, self).__init__()
        self.bn_en = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, padding=0)
        self.mode = mode
        if mode == 'G':
            self.picanet = PicanetG(size, in_channel)
        elif mode == 'L':
            self.picanet = PicanetL(in_channel)
        elif mode == 'C':
            self.picanet = None
        else:
            assert 0
        if not mode == 'C':
            self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=1, padding=0)
            self.bn_feature = nn.BatchNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)
        else:
            self.conv2 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0)

    def forward(self, *input):
        assert len(input) <= 2
        if input[1] is None:
            en = input[0]
            dec = input[0]
        else:
            en = input[0]
            dec = input[1]

        if dec.size()[2] * 2 == en.size()[2]:
            dec = F.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        elif dec.size()[2] != en.size()[2]:
            assert 0
        en = self.bn_en(en)
        en = F.relu(en)
        fmap = torch.cat((en, dec), dim=1)  # F
        fmap = self.conv1(fmap)
        fmap = F.relu(fmap)
        if not self.mode == 'C':
            fmap_att = self.picanet(fmap)  # F_att
            x = torch.cat((fmap, fmap_att), 1)
            x = self.conv2(x)
            x = self.bn_feature(x)
            dec_out = F.relu(x)
            _y = self.conv3(dec_out)
            _y = torch.sigmoid(_y)
        else:
            dec_out = self.conv2(fmap)
            _y = torch.sigmoid(dec_out)

        return dec_out, _y


class PicanetG(nn.Module):
    def __init__(self, size, in_channel):
        super(PicanetG, self).__init__()
        self.renet = Renet(size, in_channel, 100)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        size = x.size()
        # print(size)
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1)
        x = F.unfold(x, [10, 10], dilation=[3, 3])
        x = x.reshape(size[0], size[1], 10 * 10)
        kernel = kernel.reshape(size[0], 100, -1)
        x = torch.matmul(x, kernel)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x


class PicanetL(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        # print("Before unfold", x.shape)
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        # print("After unfold", x.shape)
        x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        # print(x.shape, kernel.shape)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x


class Renet(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Renet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True,
                                bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                                  bidirectional=True)  # each column
        self.conv = nn.Conv2d(512, out_channel, 1)

    def forward(self, *input):
        x = input[0]
        temp = []
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel
        for i in range(self.size):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=2)  # batch, width, height, 512
        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=3)  # batch, height, 512, width
        x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        x = self.conv(x)
        return x


if __name__ == '__main__':
    # vgg = torchvision.models.vgg16(pretrained=True)
    vgg = torchvision.models.mobilenet_v2(pretrained=True)

    device = torch.device("cuda")
    batch_size = 1
    noise = torch.randn((batch_size, 3, 224, 224)).type(torch.cuda.FloatTensor)
    target = torch.randn((batch_size, 1, 224, 224)).type(torch.cuda.FloatTensor)

    model = Unet(cfg).cuda()
    model.encoder.seq.load_state_dict(vgg.features.state_dict())
    print(model.parameters)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    print('Time: {}'.format(time.clock()))
    _, loss = model(noise, target)
#added for early stopping. The pateience calcualtes the best values of the previous 5 times.
    callbacks = [EarlyStopping(monitor='loss', patience=5)]
    model.set_callbacks(callbacks)
#added
    loss.backward()


