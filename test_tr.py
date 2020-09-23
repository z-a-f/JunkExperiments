import numpy as np
import torch
from torch import nn

torch.backends.quantized.engine = 'qnnpack'

B = 1
L = 6

iC = 1
oC = 2
# kL = 1
kH = 1
kW = 1

# strides = (1,)
# pads = (0,)
# o_pads = (0,)
# dilations = (1,)
# kernel_size = kL,

strides = (1, 1)
pads = (0, 0)
o_pads = (0, 0)
dilations = (1, 1)
groups = 1
kernel_size = kH, kW


X = torch.randn(B, iC, kH, kW).to(torch.float)
W = torch.randn(iC, oC // groups, kH, kW)

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

#################
W_prepack = torch.ops.quantized.conv_transpose2d_prepack(W_q, b, strides, pads,
                                                         o_pads, dilations,
                                                         groups)
W_unpacked, bias = torch.ops.quantized.conv_transpose2d_unpack(W_prepack)
# W_prepack = torch.ops.quantized.conv1d_prepack(W_q, b, strides, pads,
#                                                          dilations,
#                                                          groups)
# W_unpacked, bias = torch.ops.quantized.conv1d_unpack(W_prepack)

W_dq = W_unpacked.dequantize()

#################



# conv_ref = nn.Conv1d(iC, oC, kernel_size, stride=strides, padding=pads,
#                               groups=groups,
#                               bias=(b is not None), dilation=dilations)
conv_ref = nn.ConvTranspose2d(iC, oC, kernel_size, stride=strides, padding=pads,
                              groups=groups,
                              bias=(b is not None), dilation=dilations)

conv_ref.weight = nn.Parameter(W_dq)
if b is not None:
  conv_ref.bias = nn.Parameter(bias)

y_ref = conv_ref(X_dq)
y_ref_dq = torch.quantize_per_tensor(y_ref, y_s, y_zp, X_dtype).int_repr()


#################

# y_q = torch.ops.quantized.conv1d(X_q, W_prepack, y_s, y_zp).int_repr()
# y_q = torch.ops.quantized.conv_transpose2d(X_q, W_prepack, y_s, y_zp).int_repr()
y_q = torch.ops.quantized.conv_transpose2d(X_q,
                                           W_prepack,
                                           strides,
                                           pads,
                                           o_pads,
                                           dilations,
                                           groups,
                                           y_s, y_zp).int_repr()

#################

print(y_ref_dq.to(torch.float) - y_q.to(torch.float))
print((y_ref_dq == y_q).all())

print(
  f'''
    iC: {W.shape[0]}
    oC: {W.shape[1] * groups}
    k: {W.shape[2:]}
  '''
)
