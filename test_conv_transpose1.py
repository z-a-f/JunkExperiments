
import numpy as np
import torch
from torch import nn
from torch.nn import quantized as nnq

torch.backends.quantized.engine = 'qnnpack'

B = 2
L = 24

iC = 1
oC = 2
kL = 1

strides = (1,)
pads = (0,)
o_pads = (0,)
dilations = (1,)
groups = 1

X = torch.randn(B, iC, L).to(torch.float)
W = torch.randn(iC, oC // groups, kL)

X_s = 1.2
X_zp = 0
X_dtype = torch.quint8

W_s = 0.2
W_zp = 0
W_dtype = torch.qint8

b = None

y_s = 4.2
y_zp = 0

#################

X_q = torch.quantize_per_tensor(X, X_s, X_zp, X_dtype)
W_q = torch.quantize_per_tensor(W, W_s, W_zp, W_dtype)

X_dq = X_q.dequantize()
W_dq = W_q.dequantize()

#################

kernel_size = kL,

conv_ref = nn.ConvTranspose1d(iC, oC, kernel_size, stride=strides, padding=pads,
                              output_padding=o_pads, groups=groups,
                              bias=(b is not None), dilation=dilations)

conv_quant = nnq.ConvTranspose1d(iC, oC, kernel_size, stride=strides, padding=pads,
                                 output_padding=o_pads, groups=groups,
                                 bias=(b is not None), dilation=dilations)

conv_ref.weight = nn.Parameter(W_dq)
if b is not None:
  conv_ref.bias = nn.Parameter(b)

conv_quant.set_weight_bias(W_q, b)
conv_quant.scale = y_s
conv_quant.zero_point = y_zp

y_ref = conv_ref(X_dq)
y_ref_dq = torch.quantize_per_tensor(y_ref, y_s, y_zp, X_dtype).int_repr()

#################
# Check the prepack / unpack
W_prepack = torch.ops.quantized.conv_transpose1d_prepack(W_q, b, strides, pads,
                                                         o_pads, dilations,
                                                         groups)
W_unpacked, bias = torch.ops.quantized.conv_transpose1d_unpack(W_prepack)

print(W_q.dequantize() - W_unpacked.dequantize())

#################

y_q = conv_quant(X_q).int_repr()

#################

print(y_ref_dq.to(torch.float) - y_q.to(torch.float))

print(
  f'''
    iC: {W.shape[0]}
    oC: {W.shape[1] * groups}
    k: {W.shape[2:]}
  '''
)
