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

PERCEPTION = {
  "transformer_dim": 96,

}

class SwinTransformer(nn.Module):
  """
  rearrange windwos on M
  shifting boolean true or false
  relative position bias per head
  self attnention
  Attention(Q, K, V) = SoftMax(QK^T / sqrt(d) + B)V
  """
  def __init__(self, dimension:int, num_heads:int=8, window_dimension:tuple=(7, 7), apply_shift:bool=False):
    super().__init__()

    self.dimension = dimension

    vertical_windows, horizontal_windows = window_dimension
    if apply_shift:
      pass # do masking as well
    else:
      self.partition = einops_torch.Rearrange(
        "batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch -> batch (vertical_windows horizontal_windows) window",
        vertical_windows=vertical_windows,
        horizontal_windows=horizontal_windows
      )
    
    # project to learned embedding space 
    assert dimension % num_heads == 0, "Dimension must be divisible by num_heads"
    # We want to 
    #self.attention_embedding = nn.Linear(dimension, 3*dimension)
    self.embeddings = {
      "query": nn.Linear(dimension, dimension),
      "key": nn.Linear(dimension, dimension),
      "value": nn.Linear(dimension, dimension)
    }
    self.head_partition = einops_torch.Rearrange( # prepend the num_heads to transpose below
      "batch num_windows window -> batch num_heads num_windows window_head",
      num_heads=num_heads
    )
    self.attention_partition = einops_torch.Rearrange(
      "batch num_heads num_windows window_head -> batch num_windows attention"
    )

  
  def self_attention(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor):
    key_transposed = einops.rearrange(key, "batch num_heads num_windows window_head -> batch num_heads window_head num_windows")
    attention_logits = torch.matmul(query, key_transposed) / math.sqrt(self.dimension)
    attention = F.softmax(attention_logits, dim=-1)
    output = torch.matmul(attention, value)
    # do masking here?

    return attention, output


  def forward(self, x:torch.Tensor):
    windows = self.partition(x)

    modal_heads = {}
    for attention_mode in self.embeddings: # query key value
      modal_embedding = self.embeddings[attention_mode](windows)
      modal_heads[attention_mode] = self.head_partition(modal_embedding)

    attention, values = self.self_attention(modal_heads["query"], modal_heads["key"], modal_heads["value"])
    output = self.attention_partition(values)

    return output


class SwinTransformerBlock(nn.Module):
  """
  1) LN
  2) (S)W-MSA
  3) Add + Layer norm
  4) LN
  5) MLP
  6) Add + Layer norm
  """
  def __init__(self, dimension:int, feed_forward_expansion:int=4, apply_shift:bool=False):
    super().__init__()
    self.attention = nn.Sequential(
      [
        nn.LayerNorm(dimension),
        nn.SwinTransformer(dimension, 8)
      ]
    )
    self.feed_forward = nn.Sequential(
      [
        nn.LayerNorm(dimension),
        nn.Linear(dimension, feed_forward_expansion*dimension),
        nn.GELU(),
        nn.Linear(feed_forward_expansion*dimension, dimension)
      ]
    )
  
  def forward(self, patches:torch.Tensor):
    attention = patches + self.attention(patches)
    output = attention + self.feed_forward(attention)

    return output

class SwinTransformerStage(nn.Module):
  """ https://arxiv.org/pdf/2103.14030.pdf

  Consists of two swin transformer block. Reads patches in and 
  
  Window partition:
  1) Split into M x M patch windows (M x M)
  2) Shifted windows, still M x M but mask out the edges
  """
  def __init__(self, inner_dimension:int, blocks:int=2, apply_merge:bool=True, merge_reduce_factor=2):
    """ inner_dimension is the self-attention dimension, ie. C """
    super().__init__()

    input_pipeline = []
    if apply_merge:
      input_pipeline.append(
        einops_torch.Rearrange(
          "batch (vertical_patches reduce) (horizontal_patches reduce) patch -> batch vertical_patches horizontal_patches reduced_patch",
          reduce=merge_reduce_factor
        )
      )
    input_pipeline.append(nn.LazyLinear(inner_dimension))
    self.preprocess = nn.Sequential(*input_pipeline)

    # We'd like to build blocks with alternating shifted windows
    assert blocks % 2 == 0, "The number of blocks must be even to use shifted windows"
    transform_pipeline = [SwinTransformerBlock(inner_dimension, apply_shift=(index % 2 == 1)) for index in range(blocks)]
    self.transform = nn.Sequential(transform_pipeline)

  def forward(self, patches:torch.Tensor):
    patches = self.preprocess(patches)
    transformed = self.transform(patches)

    return transformed
    
class Perception(nn.Module):
  """ A SWIN Transformer perception stack

  Sequence:
  * input (H*W*1)
  * patch partition: (H/4 x W/4 x 16)
  * stage 1-2-3-4:
    1) linear embedding -> 2x swin transformer (H/4 x W/4 x C)
    2) patch merging -> 2x swin transformer (H/8 x W/8 x 2C)
    3) patch merging -> 2x swin transformer (H/16 x W/16 x 4C)
    4) patch merging -> 2x swin transformer (H/32 x W/32 x 8C)
  """
  def __init__(self, patch_dimension:tuple=(6,12)):
    super().__init__()
    # Map input image to 1d patch tokens
    vertical_patches, horizontal_patches = patch_dimension
    self.patch_partition = einops_torch.Rearrange(
      "batch 1 (vertical_patches patch_height) (horizontal_patches patch_width) -> batch vertical_patches horizontal_patches (patch_height patch_width)", 
      vertical_patches=vertical_patches, 
      horizontal_patches=horizontal_patches
    )
    self.transforms = nn.Sequential(
      [
        SwinTransformerStage(inner_dimension=CONVOLUTION_DIM, blocks=2, apply_merge=False),
        SwinTransformerStage(inner_dimension=2*CONVOLUTION_DIM, blocks=2, apply_merge=True),
        SwinTransformerStage(inner_dimension=4*CONVOLUTION_DIM, blocks=8, apply_merge=True),
        SwinTransformerStage(inner_dimension=8*CONVOLUTION_DIM, blocks=2, apply_merge=True),
      ]
    )
  
  def forward(self, images:torch.Tensor):
    patches = self.patch_partition(images)
    transformed = self.transforms(patches)

    return transformed



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