import collections

from torch import nn
from torch.utils import data as torchdata

# Our network is a constant so we define it first
def convolution(stack: collections.OrderedDict, index: int, input_channels: int, output_channels: int, use_dropout: bool):
  """ Add a convolution layer to the stack. """
  new_stack = collections.OrderedDict(stack) # I'm not gonnam mutate that chief
  new_stack[f"convolution_{index}"] = nn.Conv2d(input_channels, output_channels, kernel_size=5)

  if use_dropout:
    new_stack[f"convolution_{index}_dropout"] = nn.Dropout2d()

  new_stack[f"maxpool_{index}"] = nn.MaxPool2d(kernel_size=2)
  new_stack[f"relu_{index}"] = nn.ReLU()
  return new_stack


class View(nn.Module):
  """ from https://discuss.pytorch.org/t/how-to-build-a-view-layer-in-pytorch-for-sequential-models/53958/11 """
  def __init__(self, shape):
    super().__init__()
    self.shape = shape

  def __repr__(self):
    return f"View{self.shape}"

  def forward(self, input):
    """ Reshapes the input according to the shape saved in the view data structure. """
    batch_size = input.size(0)
    shape = (batch_size, *self.shape)
    out = input.view(shape)
    return out

STACK = collections.OrderedDict() # this is a constant and should be allcaps, I don't care what anybody says
STACK = convolution(STACK, 1, 1, 10, use_dropout=False)
STACK = convolution(STACK, 2, 10, 20, use_dropout=True)
STACK["linear_view"] = View((-1, 20*4*4))
STACK["linear_1"] = nn.Linear(20*4*4, 50)
STACK["relu_linear_1"] = nn.ReLU()
STACK["dropout_linear_1"] = nn.Dropout()
STACK["linear_2"] = nn.Linear(50, 10)
STACK["log_softmax"] = nn.LogSoftmax(dim=1)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.stack = nn.Sequential(STACK) # oh yeah that's what I'm talking about

  def forward(self, x):
    return self.stack(x)


if __name__ == "__main__":
  print('booga')