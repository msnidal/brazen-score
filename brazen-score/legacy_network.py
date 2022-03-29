import collections
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
from numpy import inner
import math

import einops
from einops.layers import torch as einops_torch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torchdata


CONVOLUTION_DIM = 62464
FEED_FORWARD_DIM = 4096


FLATTENED_SIZE = 20*70*499 # shape before flatten layer
LINEAR_NODES_1 = 2048
ATTENTION_DIM = 512 # this is supposed to be an embeddings dimension, which I think makes sense
ATTENTION_FEED_FORWARD_DIM = 2048
MAX_LENGTH = 75
SYMBOLS_DIM = 758



TRANSFORMER_DIM = 32




class ConvolutionNet(nn.Module):
  def __init__(self, input_channels:int, output_channels:int, kernel_size:int=3, stride:int=1, padding:int=0, use_pooling:bool=False):
    super().__init__()
    self.convolution = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=3, stride=2) if use_pooling else None

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    out = self.convolution(x)
    out = self.relu(out)
    if self.pool:
      out = self.pool(out)

    return out


class SelfAttentionStack(nn.Module):
  def __init__(self, width, inner_feed_forward_width, num_heads=8):
    super().__init__()
    self.query = nn.Linear(width, width)
    self.key = nn.Linear(width, width)
    self.value = nn.Linear(width, width)

    self.attention = nn.MultiheadAttention(width, num_heads)
    self.layer_norm = nn.LayerNorm(width)
    self.linear_1 = nn.Linear(width, inner_feed_forward_width)
    self.relu = nn.ReLU()
    self.linear_2 = nn.Linear(inner_feed_forward_width, width)
  
  def forward(self, x) -> torch.Tensor:
    # multihead attention block
    out, weights = self.attention(self.query(x), self.key(x), self.value(x))
    attention_out = self.layer_norm(out + x) # add and norm at last step. Verify they're not teh same? 

    # feed forward block
    out = self.linear_1(attention_out)
    out = self.relu(out)
    out = self.linear_2(out)
    out = self.layer_norm(out + attention_out)

    return out


class PositionalEncoding(nn.Module):
  """ Credit: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
  Paper has more details https://arxiv.org/pdf/1706.03762.pdf 
  """
  def __init__(self, model_dim: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
    positional_encoding = torch.zeros(max_len, 1, model_dim)
    positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('positional_encoding', positional_encoding) # TODO: make more use of this https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/7

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """ Args: x: Tensor, shape [seq_len, batch_size, embedding_dim] """
    x += self.positional_encoding[:x.size(0)]

    return self.dropout(x)

class BrazenNet(nn.Module):
  def __init__(self, dropout = 0.1):
    super().__init__()
    # AlexNet 
    self.convolution_stack = nn.Sequential(
      ConvolutionNet(input_channels=1, output_channels=32, kernel_size=11, stride=4, padding=2, use_pooling=True),
      ConvolutionNet(input_channels=32, output_channels=96, kernel_size=5, padding=2, use_pooling=True),
      ConvolutionNet(input_channels=96, output_channels=192, kernel_size=3, padding=1),
      ConvolutionNet(input_channels=192, output_channels=128, kernel_size=3, padding=1),
      ConvolutionNet(input_channels=128, output_channels=128, kernel_size=3, padding=1, use_pooling=True)
    )
    self.convolution_pool = nn.AvgPool2d((6, 6))
    self.flatten = nn.Flatten() # start dim = 1

    # Feed forward
    self.feed_forward_stack = nn.Sequential(
      nn.Linear(CONVOLUTION_DIM, FEED_FORWARD_DIM),
      nn.ReLU(inplace=True),
      nn.Linear(FEED_FORWARD_DIM, FEED_FORWARD_DIM),
      nn.ReLU(inplace=True),
      nn.Linear(FEED_FORWARD_DIM, ATTENTION_DIM)
    )

    # Self attention
    self.positional_encoding = PositionalEncoding(ATTENTION_DIM, max_len=MAX_LENGTH)
    self.attention_stack = nn.Sequential(
      SelfAttentionStack(ATTENTION_DIM, ATTENTION_FEED_FORWARD_DIM),
      SelfAttentionStack(ATTENTION_DIM, ATTENTION_FEED_FORWARD_DIM),
      nn.Linear(ATTENTION_DIM, SYMBOLS_DIM + 1)
    )

    self.softmax = nn.LogSoftmax(dim=-1)


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Convolution
    out = self.convolution_stack(x)
    out = self.flatten(out)

    # Linear
    out = self.feed_forward_stack(out)

    # Positional encodings
    out = out.unsqueeze(1)
    out = out.repeat(1, MAX_LENGTH, 1)
    out = self.positional_encoding(out)

    # Self-attention
    out = self.attention_stack(out)
    out = self.softmax(out)

    return out