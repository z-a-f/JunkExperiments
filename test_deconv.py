#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F

from torch.testing._internal.common_quantized import override_quantized_engine

class DecoderModel(nn.Module):
    def __init__(self):
        super(DecoderModel, self).__init__()
        self.linear_input = nn.Linear(2, 36)
        self.deconv1 = nn.ConvTranspose1d(1, 7, 1)  # Unsqueeze
        self.act1 = nn.ELU()
        self.deconv2 = nn.ConvTranspose1d(7, 5, 3, stride=3, output_padding=2)
        self.act2 = nn.ELU()
        self.deconv3 = nn.ConvTranspose1d(5, 3, 5, stride=3, output_padding=1)
        self.act3 = nn.ELU()
        self.deconv4 = nn.ConvTranspose1d(3, 1, 7, stride=3, padding=1)
        self.act4 = nn.ELU()

        self.flatten = nn.Flatten()
        self.linear_output = nn.Linear(1001, 1001)

    def forward(self, x):
        x = self.linear_input(x)
        x.unsqueeze_(1)  # Unflatten
        x = self.deconv1(x)
        x = self.act1(x)
        x = self.deconv2(x)
        x = self.act2(x)
        x = self.deconv3(x)
        x = self.act3(x)
        x = self.deconv4(x)
        x = self.act4(x)
        x = self.flatten(x)
        x = self.linear_output(x)
        return x

decode_model = DecoderModel().cpu().eval()
print(decode_model.deconv1.weight.shape)

with override_quantized_engine('qnnpack'):
  decode_model.qconfig = torch.quantization.default_qconfig
  torch.quantization.prepare(decode_model, inplace=True)
  qdecode_model = torch.quantization.convert(decode_model, inplace=False)
  print(qdecode_model)
