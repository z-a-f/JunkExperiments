import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.quantized import functional as qF

x = np.array([[[[1.]],
               [[1.]]]], dtype=np.float32)
s = 1.0
z = 0

output_shape = (1, 1, 1)

tx = torch.from_numpy(x)
qx = torch.quantize_per_tensor(tx, s, z, torch.quint8)
dqx = qx.dequantize()

y = F.adaptive_avg_pool3d(dqx, output_shape)
qy = qF.adaptive_avg_pool3d(qx, output_shape)

print(x.shape)
print(output_shape)
print(y.shape)
print(qy.shape)

print(y)
print(qy)
