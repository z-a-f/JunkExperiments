
import numpy as np
import torch
from torch import nn

torch.backends.quantized.engine = 'qnnpack'

X_np = np.array([[[[1.2000, 2.4000, 2.4000, 2.4000, 1.2000, 0.0000, 1.2000],
                   [2.4000, 1.2000, 3.6000, 0.0000, 2.4000, 3.6000, 0.0000],
                   [0.0000, 3.6000, 3.6000, 1.2000, 2.4000, 2.4000, 0.0000],
                   [0.0000, 2.4000, 1.2000, 0.0000, 2.4000, 1.2000, 1.2000],
                   [0.0000, 3.6000, 0.0000, 1.2000, 2.4000, 0.0000, 1.2000],
                   [0.0000, 2.4000, 2.4000, 3.6000, 0.0000, 0.0000, 1.2000],
                   [3.6000, 0.0000, 3.6000, 0.0000, 3.6000, 0.0000, 1.2000],
                   [3.6000, 0.0000, 2.4000, 0.0000, 3.6000, 0.0000, 0.0000],
                   [3.6000, 3.6000, 0.0000, 0.0000, 2.4000, 3.6000, 3.6000],
                   [0.0000, 1.2000, 2.4000, 1.2000, 0.0000, 1.2000, 1.2000]],

                  [[1.2000, 1.2000, 2.4000, 2.4000, 1.2000, 3.6000, 3.6000],
                   [3.6000, 0.0000, 3.6000, 3.6000, 3.6000, 1.2000, 1.2000],
                   [1.2000, 3.6000, 2.4000, 3.6000, 3.6000, 2.4000, 1.2000],
                   [1.2000, 0.0000, 3.6000, 0.0000, 0.0000, 1.2000, 0.0000],
                   [3.6000, 0.0000, 0.0000, 2.4000, 0.0000, 2.4000, 2.4000],
                   [3.6000, 2.4000, 3.6000, 1.2000, 1.2000, 1.2000, 2.4000],
                   [3.6000, 0.0000, 3.6000, 0.0000, 0.0000, 3.6000, 2.4000],
                   [0.0000, 2.4000, 1.2000, 2.4000, 2.4000, 2.4000, 3.6000],
                   [1.2000, 1.2000, 2.4000, 2.4000, 2.4000, 1.2000, 3.6000],
                   [0.0000, 3.6000, 1.2000, 0.0000, 2.4000, 3.6000, 0.0000]]]])
X = torch.randn(X_np.shape)
X_s = 1.2
X_zp = 0
X_dtype = torch.quint8

W_np = np.array([[[[ 0.8000]],
                  [[ 0.0000]],
                  [[ 0.4000]],
                  [[ 0.2000]]],
                 [[[ 0.0000]],
                  [[ 0.8000]],
                  [[-1.0000]],
                  [[ 0.4000]]]])
W_s = 0.2
W_zp = 0
W_dtype = torch.qint8

b = None

strides = (1, 1)
pads = (0, 0)
o_pads = (0, 0)
dilations = (1, 1)
groups = 2

y_s = 4.2
y_zp = 0

#################

X = torch.from_numpy(X_np).to(torch.float)
W = torch.from_numpy(W_np).to(torch.float)

X_q = torch.quantize_per_tensor(X, X_s, X_zp, X_dtype)
W_q = torch.quantize_per_tensor(W, W_s, W_zp, W_dtype)

X_dq = X_q.dequantize()
W_dq = W_q.dequantize()

print(f'Input shape: {X.shape}, Weight shape: {W.shape}')

#################

iC = W_dq.shape[0]
oC = W_dq.shape[1] * groups
kernel_size = W_dq.shape[2:]

conv_ref = nn.ConvTranspose2d(iC, oC, kernel_size, stride=strides, padding=pads,
                              output_padding=o_pads, groups=groups,
                              bias=(b is not None), dilation=dilations)

conv_ref.weight = nn.Parameter(W_dq)
if b is not None:
  conv_ref.bias = nn.Parameter(b)

y_ref = conv_ref(X_dq)
y_ref_dq = torch.quantize_per_tensor(y_ref, y_s, y_zp, X_dtype).dequantize()

#################
W_prepack = torch.ops.quantized.conv_transpose2d_prepack(W_q, b, strides, pads,
                                                         o_pads, dilations,
                                                         groups)
W_unpacked, bias = torch.ops.quantized.conv_transpose2d_unpack(W_prepack)

print(W_q.dequantize() - W_unpacked.dequantize())


#################

y_q = torch.ops.quantized.conv_transpose2d(X_q, W_prepack, y_s, y_zp).dequantize()

#################

print(y_ref_dq.to(torch.float) - y_q.to(torch.float))


print(
  f'''
    iC: {W.shape[0]}
    oC: {W.shape[1] * groups}
    k: {W.shape[2:]}
  '''
)
