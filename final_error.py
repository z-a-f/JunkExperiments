import torch

try:
  from typing_extensions import Final
except ImportError:
  from torch.jit import Final

class Foo(torch.nn.Module):
  _my_class_variable: Final[int] = 42
  def __init__(self):
    super(Foo, self).__init__()
    self.my_other_variable = 100 + self._my_class_variable
  def forward(self, x):
    print(self._my_class_variable)
    return x + self.my_other_variable

if __name__ == '__main__':
  m = torch.jit.script(Foo())
  # print(m)
  print(m.graph)

  ten = torch.tensor([1, 2, 3])
  print(m(ten))
