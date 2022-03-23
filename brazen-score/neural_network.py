import collections
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
from numpy import inner
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torchdata


CONV_CHANNELS_1 = 10
CONV_CHANNELS_2 = 20
FLATTENED_SIZE = 20*23*123 # idk where this comes from LMAO got it from the shape *dab*
LINEAR_NODES_1 = 1024
ATTENTION_DIM = 256 # this is supposed to be an embeddings dimension, which I think makes sense
ATTENTION_FEED_FORWARD_DIM = 1024
MAX_LENGTH = 75
SYMBOLS_DIM = 756


class ConvolutionStack(nn.Module):
  def __init__(self, input_channels: int, output_channels: int, dropout=0.1):
    super().__init__()
    self.convolution = nn.Conv2d(input_channels, output_channels, kernel_size=5, padding=1)
    self.dropout = nn.Dropout(dropout)
    self.pool = nn.MaxPool2d(kernel_size=2)
    self.relu = nn.ReLU()

  def forward(self, x) -> torch.Tensor:
    out = self.convolution(x)
    out = self.dropout(out)
    out = self.pool(out)
    out = self.relu(out)

    return out


class SelfAttentionStack(nn.Module):
  def __init__(self, width, inner_feed_forward_width, num_heads=8):
    super().__init__()
    self.attention = nn.MultiheadAttention(width, num_heads)
    self.layer_norm = nn.LayerNorm(width)
    self.linear_1 = nn.Linear(width, inner_feed_forward_width)
    self.relu = nn.ReLU()
    self.linear_2 = nn.Linear(inner_feed_forward_width, width)
  
  def forward(self, x) -> torch.Tensor:
    # multihead attention block
    out, weights = self.attention(x, x, x)
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
    self.convolution_1 = ConvolutionStack(1, CONV_CHANNELS_1, False)
    self.convolution_2 = ConvolutionStack(CONV_CHANNELS_1, CONV_CHANNELS_2, True)
    self.flatten = nn.Flatten() # start dim = 1
    self.linear_1 = nn.Linear(FLATTENED_SIZE, LINEAR_NODES_1)
    self.linear_relu = nn.ReLU()
    self.linear_dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(LINEAR_NODES_1, ATTENTION_DIM)
    # need to run the model for 75 symbols, so we need to pad the input
    self.positional_encoding = PositionalEncoding(ATTENTION_DIM, max_len=MAX_LENGTH)
    self.attention = SelfAttentionStack(ATTENTION_DIM, ATTENTION_FEED_FORWARD_DIM)
    self.linear_out = nn.Linear(ATTENTION_DIM, SYMBOLS_DIM + 1) # plus one is for empty
    self.softmax = nn.LogSoftmax(dim=-1)


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.convolution_1(x)
    out = self.convolution_2(out)
    out = self.flatten(out)
    out = self.linear_1(out)
    out = self.linear_relu(out)
    out = self.linear_dropout(out)
    out = self.linear_2(out)
    
    # We need to run the model for max length
    out = out.unsqueeze(1)
    out = out.repeat(1, MAX_LENGTH, 1)
    out = self.positional_encoding(out)
    out = self.attention(out)
    out = self.linear_out(out)
    out = self.softmax(out)

    return out # TODO: batch = all encodings. eze the output before the attention net? Run several times with same frozen values? Or just run with a different positional embedding.