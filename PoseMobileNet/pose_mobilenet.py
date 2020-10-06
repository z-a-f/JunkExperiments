import os
import logging
import pdb

import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary

from torch.quantization import QuantStub, DeQuantStub

logger = logging.getLogger(__name__)


# return an interger number, just need to make sure that every channel is divisible by 8
def _make_divisible(v, divisor = 8, min_value = None):
    """
    This function is taken from the original tf repo.
    Reference:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    v = v * 1.0
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# Depthwise Conv + BatchNorm + ReLU block
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True)
        )

# Residual block
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer = None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

        # special layer for quantization only
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            # (1) Use this for normal training and testing
            #return x + self.conv(x)
            # (2) Use this when do quantization
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)



class PoseMobileNet(nn.Module):
    def __init__(self, **kwargs):
        super(PoseMobileNet, self).__init__()
        # 0. Initialize
        res_block = InvertedResidual
        inverted_residual_setting = [
                # ratio, channel, repeat n_times, stride
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        self.input_channel = 3
        self.output_channel = 32
        self.output_channel = _make_divisible(self.output_channel)
        self.last_channel = 1280
        self.last_channel = _make_divisible(self.last_channel)

        # 1. Mobilenet v2 backbone
        # https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
        features = [ConvBNReLU(self.input_channel, self.output_channel, stride=2)]
        self.input_channel = self.output_channel
        for t, c, n, s in inverted_residual_setting:
            self.output_channel = _make_divisible(c)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(res_block(self.input_channel, self.output_channel, stride, expand_ratio=t))
                self.input_channel = self.output_channel

        features.append(ConvBNReLU(self.input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.output_channel = self.last_channel

        # 2. Conv transpose layers
        self.conv_transpose_layers = self._make_conv_transpose_layer(
            num_layers = 3,
            filters = [128, 128, 128],
            kernals = [4, 4, 4]
        )

        # 3. Final conv layer
        self.final_layer = nn.Conv2d(
            in_channels = self.output_channel,
            out_channels = 21, # 21 keypoints
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        # 4. Two special layers for quantization
        self.quant = QuantStub()
        self.dequant = DeQuantStub()


    def _make_conv_transpose_layer(self, num_layers, filters, kernals):
        # num_layers = 3
        # kernals = [4, 4, 4]
        # filters = [128, 128, 128]
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels = self.output_channel,
                    out_channels = filters[i],
                    kernel_size = kernals[i],
                    stride = 2,
                    padding = 1,
                    output_padding = 0,
                    bias = False))
            layers.append(nn.BatchNorm2d(num_features = filters[i]))
            layers.append(nn.ReLU(inplace = True))
            self.output_channel = filters[i]

        return nn.Sequential(*layers)

    def forward(self, x):
        # special layer for quantization
        x = self.quant(x)

        #============================
        x = self.features(x)
        x = self.conv_transpose_layers(x)
        x = self.final_layer(x)
        #============================

        # special layer for quantization
        x = self.dequant(x)

        return x

    def init_weights(self, pretrained=''):
        pass

    # Make sure that no ReLU6 in model. It can't be fused, Use ReLU instead.
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)



def get_pose_net(is_train,  **kwargs):
    model = PoseMobileNet(**kwargs)

    if is_train:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


if __name__ == '__main__':
    net = get_pose_net(is_train = False)
    summary(net.cuda(), (3, 192, 256))  # input shape = [C, H, W]
