
import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 3
CHANNELS = 2
Hin = 11
Win = 17

PADDING = 5
QTYPE = torch.qint8

def get_qparams(inp, qtype):
  fmin = inp.min().item()
  fmax = inp.max().item()
  qinfo = torch.iinfo(qtype)
  qmin = qinfo.min
  qmax = qinfo.max
  if fmin == fmax == 0:
    return 0.0, 0  # Scale, Zero Point
  fmin = min(fmin, 0)
  fmax = max(fmax, 0)
  scale = (fmax - fmin) / (qmax - qmin)
  zp = round(qmin - fmin / scale)
  return scale, zp

def test_padding(inp):
  padder = nn.ReflectionPad1d(PADDING)
  y = padder(inp)
  print(f'x: {inp[0]}')
  print(f'y: {y[0]}')
  print(f'x.shape: {inp.shape}')
  print(f'y.shape: {y.shape}')
  return y


if __name__ == '__main__':
  # 1D tests
  # x = torch.randn((BATCH_SIZE, CHANNELS, Win))
  x = torch.arange(BATCH_SIZE * CHANNELS * Win).to(torch.float)
  x = x.reshape(BATCH_SIZE, CHANNELS, Win)
  s, z = get_qparams(x, QTYPE)
  qx = torch.quantize_per_tensor(x, scale=s, zero_point=z, dtype=QTYPE)
  qx.copy_(qx)
  # qx.resize_(CHANNELS * BATCH_SIZE * Win)
  test_padding(x)
  test_padding(qx)
