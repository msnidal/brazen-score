import collections
import math
from turtle import position

import einops
from einops.layers import torch as einops_torch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torchdata


PATCH_DIM = 16
NUM_HEADS = 8

class SwinTransformer(nn.Module):
  """ Implements locality self attention from https://arxiv.org/pdf/2112.13492.pdf
  Attention(Q, K, V) = SoftMax(QK^T / sqrt(d) + B)V
  """
  def __init__(self, dimension:int, num_heads:int=8, window_dimension:tuple=(7, 7), apply_shift:bool=False):
    super().__init__()

    self.dimension = dimension

    vertical_windows, horizontal_windows = window_dimension
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

    # do this in 2n - 1 and project
    position_bias_dim = dimension ** 2
    self.position_bias = nn.parameter.Parameter(torch.randn(num_heads, position_bias_dim, position_bias_dim))

    if apply_shift:
      self.mask = torch.zeros(num_heads, vertical_windows, horizontal_windows, dtype=torch.bool)
    self.self_attention_mask = {
      "mask": torch.eye(dimension, dtype=torch.bool),
      "fill": float('-inf')
    }
    einops.rearrange(self.self_attention_mask["mask"], "(window_height window_width) (next_window_height next_window_width) -> window_height window_width next_window_height next_window_width")
    # einops reduce self.attention_mask["mask"] and mask out teh bottom right corner as well as right and bottom aginst eachother




    self.head_partition = einops_torch.Rearrange( # prepend the num_heads to transpose below
      "batch num_windows window -> batch num_heads num_windows window_head",
      num_heads=num_heads
    )
    self.attention_partition = einops_torch.Rearrange(
      "batch num_heads num_windows window_head -> batch num_windows attention"
    )

  
  def self_attention(self, windows:torch.Tensor):
    """ Window pass through to the output
    """

    heads = {}
    for attention_mode in self.embeddings: # query key value
      modal_embedding = self.embeddings[attention_mode](windows)
      heads[attention_mode] = self.head_partition(modal_embedding)
    query, key, value = heads["query"], heads["key"], heads["value"]


    key_transposed = einops.rearrange(key, "batch num_heads num_windows window_head -> batch num_heads window_head num_windows")
    attention_logits = (torch.matmul(query, key_transposed) / math.sqrt(self.dimension)) + self.position_bias
    attention = F.softmax(attention_logits, dim=-1)
    attention.masked_fill_(self.self_attention_mask["mask"], self.self_attention_mask["fill"])

    self_attention = torch.matmul(attention, value)

    return self_attention


  def forward(self, x:torch.Tensor):
    windows = self.partition(x)
    self_attention = self.self_attention(windows)
    output = self.attention_partition(self_attention)

    return output


class SwinTransformerBlock(nn.Module):
  """ Applies a single layer of the transformer to the input
  """
  def __init__(self, dimension:int, window_dim:8, feed_forward_expansion:int=4, apply_shift:bool=False):
    super().__init__()
    self.apply_shift = window_dim if apply_shift is True else None

    self.attention = nn.Sequential(
      [
        nn.LayerNorm(dimension),
        nn.SwinTransformer(dimension, window_dim=window_dim, apply_shift=apply_shift),
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
    if self.apply_shift:
      patches.roll(self.apply_shift, dims=(2, 3))

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
  def __init__(self, patch_width, patch_height):
    super().__init__()
    # Map input image to 1d patch tokens
    self.patch_partition = einops_torch.Rearrange(
      "batch 1 (vertical_patches patch_height) (horizontal_patches patch_width) -> batch vertical_patches horizontal_patches (patch_height patch_width)", 
      patch_height=patch_height, 
      patch_width=patch_width
    )

    inner_patch_dim = patch_width * patch_height
    self.transforms = nn.Sequential(
      [
        SwinTransformerStage(inner_dimension=inner_patch_dim, blocks=2, apply_merge=False),
        SwinTransformerStage(inner_dimension=2*inner_patch_dim, blocks=2, apply_merge=True),
        SwinTransformerStage(inner_dimension=4*inner_patch_dim, blocks=8, apply_merge=True),
        SwinTransformerStage(inner_dimension=8*inner_patch_dim, blocks=2, apply_merge=True),
      ]
    )
  
  def forward(self, images:torch.Tensor):
    patches = self.patch_partition(images)
    transformed = self.transforms(patches)

    return transformed
