import collections
import math
from turtle import position
from typing import OrderedDict

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
  def __init__(self, dimension:int, num_heads:int=8, window_dim:int=8, apply_shift:bool=False):
    super().__init__()

    self.dimension = dimension

    assert dimension % num_heads == 0, "Dimension must be divisible by num_heads"
    self.embeddings = {
      "query": nn.Linear(dimension, dimension),
      "key": nn.Linear(dimension, dimension),
      "value": nn.Linear(dimension, dimension)
    }

    position_bias_dim = (2 * dimension - 1)
    position_bias = nn.parameter.Parameter(torch.randn(position_bias_dim, position_bias_dim))
    #self.position_bias = position_bias.view(dimension**2, dimension**2) # TODO: verify relative position bias
    self.position_bias = torch.zeros((dimension, dimension))

    if apply_shift:
      self.mask = torch.zeros(num_heads, window_dim, window_dim, dtype=torch.bool)
    self.self_attention_mask = {
      "mask": torch.eye(dimension, dtype=torch.bool),
      "fill": float('-inf')
    }
    #einops.rearrange(
    #  self.self_attention_mask["mask"], 
    #  "(window_height window_width) (next_window_height next_window_width) -> window_height window_width next_window_height next_window_width"
    #)
    # einops reduce self.attention_mask["mask"] and mask out teh bottom right corner as well as right and bottom aginst eachother

    self.head_partition = einops_torch.Rearrange(
      "batch num_windows (num_heads embedding) -> batch num_heads num_windows embedding",
      num_heads=num_heads
    )
    self.attention_partition = einops_torch.Rearrange(
      "batch num_heads num_windows embedding -> batch num_windows (num_heads embedding)"
    )

  
  def self_attention(self, windows:torch.Tensor):
    """ Window pass through to the output
    """

    heads = {}
    for attention_mode in self.embeddings: # query key value
      modal_embedding = self.embeddings[attention_mode](windows)
      heads[attention_mode] = self.head_partition(modal_embedding)
    query, key, value = heads["query"], heads["key"], heads["value"]


    key_transposed = einops.rearrange(key, "batch num_heads num_windows window -> batch num_heads window num_windows")
    # TODO: Q, K, V element of R (M^2 + d), right now is (batch num_heads num_windows window) where window is an embedding projection into 128 out of 1024
    attention_logits = (torch.matmul(query, key_transposed) / math.sqrt(self.dimension)) + self.position_bias
    attention = F.softmax(attention_logits, dim=-1)
    attention.masked_fill_(self.self_attention_mask["mask"], self.self_attention_mask["fill"])

    self_attention = torch.matmul(attention, value)

    return self_attention


  def forward(self, windows:torch.Tensor):
    self_attention = self.self_attention(windows)
    output = self.attention_partition(self_attention)

    return output


class SwinTransformerBlock(nn.Module):
  """ Applies a single layer of the transformer to the input
  """
  def __init__(self, embedding_dim:int, window_dim:int=8, feed_forward_expansion:int=4, apply_shift:bool=False):
    super().__init__()
    self.apply_shift = window_dim / 2 if apply_shift is True else None
    self.attention_dim = embedding_dim * window_dim * window_dim

    self.partition = einops_torch.Rearrange(
      "batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) embedding -> batch (vertical_windows horizontal_windows) (vertical_patches horizontal_patches embedding)",
      vertical_patches=window_dim,
      horizontal_patches=window_dim
    )
    
    attention = OrderedDict()
    attention["layer_norm"] = nn.LayerNorm(self.attention_dim)
    attention["transform"] = SwinTransformer(self.attention_dim, apply_shift=apply_shift, window_dim=window_dim)

    self.attention = nn.Sequential(attention)

    feed_forward = OrderedDict()
    feed_forward["layer_norm"] = nn.LayerNorm(self.attention_dim)
    feed_forward["linear_1"] = nn.Linear(self.attention_dim, self.attention_dim * feed_forward_expansion)
    feed_forward["gelu"] = nn.GELU()
    feed_forward["linear_2"] = nn.Linear(self.attention_dim * feed_forward_expansion, self.attention_dim)
    self.feed_forward = nn.Sequential(feed_forward)
  
  def forward(self, embeddings:torch.Tensor):
    if self.apply_shift:
      embeddings.roll(self.apply_shift, dims=(2, 3))

    windows = self.partition(embeddings)
    attention = windows + self.attention(windows)
    output = attention + self.feed_forward(attention)

    return output

class SwinTransformerStage(nn.Module):
  """ https://arxiv.org/pdf/2103.14030.pdf

  Consists of two swin transformer block. Reads patches in and 
  
  Window partition:
  1) Split into M x M patch windows (M x M)
  2) Shifted windows, still M x M but mask out the edges
  """
  def __init__(self, embedding_dim:int, blocks:int=2, apply_merge:bool=True, merge_reduce_factor=2):
    """ inner_dimension is the self-attention dimension, ie. C """
    super().__init__()

    input_pipeline = OrderedDict()
    if apply_merge:
      input_pipeline["reduce"] = einops_torch.Rearrange(
        "batch (vertical_patches vertical_reduce) (horizontal_patches horizontal_reduce) patch -> batch vertical_patches horizontal_patches (vertical_reduce horizontal_reduce patch)",
        vertical_reduce=merge_reduce_factor,
        horizontal_reduce=merge_reduce_factor
      )
    input_pipeline["embed"] = nn.LazyLinear(embedding_dim)
    self.preprocess = nn.Sequential(input_pipeline)

    # We'd like to build blocks with alternating shifted windows
    assert blocks % 2 == 0, "The number of blocks must be even to use shifted windows"
    transform_pipeline = OrderedDict()
    for index in range(blocks):
      is_odd = index % 2 == 1
      transform_pipeline[f"block_{index}"] = SwinTransformerBlock(embedding_dim=embedding_dim, apply_shift=is_odd)

    self.transform = nn.Sequential(transform_pipeline)

  def forward(self, patches:torch.Tensor):
    embeddings = self.preprocess(patches)
    transformed = self.transform(embeddings)

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
  def __init__(self, patch_width=4, patch_height=4):
    super().__init__()
    # Map input image to 1d patch tokens
    self.patch_partition = einops_torch.Rearrange(
      "batch 1 (vertical_patches patch_height) (horizontal_patches patch_width) -> batch vertical_patches horizontal_patches (patch_height patch_width)", 
      patch_height=patch_height, 
      patch_width=patch_width
    )

    patch_dim = patch_width * patch_height
    transforms = OrderedDict()
    transforms["stage_1"] = SwinTransformerStage(patch_dim, blocks=2, apply_merge=False)
    transforms["stage_2"] = SwinTransformerStage(2*patch_dim, blocks=2, apply_merge=True)
    transforms["stage_3"] = SwinTransformerStage(4*patch_dim, blocks=8, apply_merge=True)
    transforms["stage_4"] = SwinTransformerStage(8*patch_dim, blocks=2, apply_merge=True)
    self.transforms = nn.Sequential(transforms)
  
  def forward(self, images:torch.Tensor):
    patches = self.patch_partition(images)
    transformed = self.transforms(patches)

    return transformed
