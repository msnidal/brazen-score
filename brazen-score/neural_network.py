import collections
import math
from turtle import position
from typing import OrderedDict

import einops
from einops.layers import torch as einops_torch
import torch
from torch import nn
from torch.utils import data as torchdata


TRANSFORMER_DIM = 128
NUM_PATCHES = 567
PATCH_DIM = 16
NUM_HEADS = 8


class SwinTransformer(nn.Module):
    """Implements locality self attention from https://arxiv.org/pdf/2112.13492.pdf
    Attention(Q, K, V) = SoftMax(QK^T / sqrt(d) + B)V
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        window_dim: int = 8,
        apply_shift: bool = False,
    ):
        super().__init__()

        self.dimension = embedding_dim

        assert (
            embedding_dim % num_heads == 0
        ), "Dimension must be divisible by num_heads"
        self.embeddings = {
            "query": nn.Linear(embedding_dim, embedding_dim),
            "key": nn.Linear(embedding_dim, embedding_dim),
            "value": nn.Linear(embedding_dim, embedding_dim),
        }

        position_bias_dim = window_dim**2  # (window_dim * 2) - 1
        position_bias = nn.parameter.Parameter(
            torch.Tensor(size=(num_heads, position_bias_dim, position_bias_dim))
        )
        self.position_bias = nn.init.trunc_normal_(position_bias, mean=0, std=0.02)
        # self.position_bias = self.relative_embedding(normalized_bias, window_dim)

        # num_windows = 160 # TODO: figure this out
        # if apply_shift:
        #  self.mask = torch.zeros(num_windows, window_dim**2, embedding_dim, dtype=torch.bool)
        # self.self_attention_mask = {
        #  "mask": torch.eye(embedding_dim, dtype=torch.bool),
        #  "fill": float('-inf')
        # }
        # einops.rearrange(
        #  self.self_attention_mask["mask"],
        #  "(window_height window_width) (next_window_height next_window_width) -> window_height window_width next_window_height next_window_width"
        # )
        # einops reduce self.attention_mask["mask"] and mask out teh bottom right corner as well as right and bottom aginst eachother

        self.head_partition = einops_torch.Rearrange(
            "batch num_windows num_patches (num_heads embedding) -> batch num_windows num_heads num_patches embedding",
            num_heads=num_heads,
        )
        self.attention_partition = einops_torch.Rearrange(
            "batch num_windows num_heads num_patches embedding -> batch num_windows num_patches (num_heads embedding)"
        )

    def self_attention(self, windows: torch.Tensor):
        """Window pass through to the output"""

        heads = {}
        for attention_mode in self.embeddings:  # query key value
            modal_embedding = self.embeddings[attention_mode](windows)
            heads[attention_mode] = self.head_partition(modal_embedding)
        query, key, value = heads["query"], heads["key"], heads["value"]

        key_transposed = einops.rearrange(
            key,
            "batch num_windows num_heads num_patches embedding -> batch num_windows num_heads embedding num_patches",
        )
        attention_logits = (
            torch.matmul(query, key_transposed) / math.sqrt(self.dimension)
        ) + self.position_bias  # + self.mask
        attention = nn.functional.softmax(attention_logits, dim=-1)
        # attention.masked_fill_(self.self_attention_mask["mask"], self.self_attention_mask["fill"])

        self_attention = torch.matmul(attention, value)

        return self_attention

    def forward(self, windows: torch.Tensor):
        self_attention = self.self_attention(windows)
        output = self.attention_partition(self_attention)

        return output


class SwinTransformerBlock(nn.Module):
    """Applies a single layer of the transformer to the input
    Note: window_dim is the number of patches per window
    """

    def __init__(
        self,
        embedding_dim: int,
        patch_dim: int,
        apply_shift: bool = False,
        window_dim: int = 8,
        feed_forward_expansion: int = 4,
        image_shape: tuple = (2048, 320),
    ):
        super().__init__()
        self.apply_shift = window_dim / 2 if apply_shift is True else None

        self.partition = einops_torch.Rearrange(
            "batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch -> batch (vertical_windows horizontal_windows) (vertical_patches horizontal_patches) patch",
            vertical_patches=window_dim,
            horizontal_patches=window_dim,
        )  # fix the number of patches per window, let it find number of windows from image
        self.departition = einops_torch.Rearrange(
            "batch (vertical_windows horizontal_windows) (vertical_patches horizontal_patches) patch -> batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch",
            vertical_windows=image_shape[1] / patch_dim / window_dim,
            horizontal_windows=image_shape[0] / patch_dim / window_dim,
            vertical_patches=window_dim,
            horizontal_patches=window_dim,
        )

        attention = OrderedDict()
        attention["layer_norm"] = nn.LayerNorm(embedding_dim)
        attention["transform"] = SwinTransformer(
            embedding_dim, apply_shift=apply_shift, window_dim=window_dim
        )
        self.attention = nn.Sequential(attention)

        feed_forward = OrderedDict()
        feed_forward["layer_norm"] = nn.LayerNorm(embedding_dim)
        feed_forward["linear_1"] = nn.Linear(
            embedding_dim, embedding_dim * feed_forward_expansion
        )
        feed_forward["gelu"] = nn.GELU()
        feed_forward["linear_2"] = nn.Linear(
            embedding_dim * feed_forward_expansion, embedding_dim
        )
        self.feed_forward = nn.Sequential(feed_forward)

    def forward(self, embeddings: torch.Tensor):
        if self.apply_shift:
            embeddings = embeddings.roll(
                self.apply_shift, dims=(1, 2)
            )  # TODO: explore einops.reduce

        patches = self.partition(embeddings)
        attention = patches + self.attention(patches)
        output = attention + self.feed_forward(attention)
        output_patches = self.departition(output)

        return output_patches


class SwinTransformerStage(nn.Module):
    """Two sequential swin transformer blocks. The second block uses a shifted window for context
    https://arxiv.org/pdf/2103.14030.pdf
    """

    def __init__(
        self,
        embedding_dim: int,
        num_blocks: int = 2,
        apply_merge: bool = True,
        patch_dim: int = 8,
        image_shape: tuple = (2048, 320),
        merge_reduce_factor=2,
    ):
        super().__init__()

        input_pipeline = OrderedDict()
        if apply_merge:
            # Reduce patches - merge by the reduction factor to 1, widen the embeddings
            input_pipeline["reduce"] = einops_torch.Rearrange(
                "batch (vertical_patches vertical_reduce) (horizontal_patches horizontal_reduce) patch-> batch vertical_patches horizontal_patches (vertical_reduce horizontal_reduce patch)",
                vertical_reduce=merge_reduce_factor,
                horizontal_reduce=merge_reduce_factor,
            )
            embedding_dim *= merge_reduce_factor
            patch_dim *= merge_reduce_factor

        input_pipeline["embed"] = nn.LazyLinear(embedding_dim)
        self.preprocess = nn.Sequential(input_pipeline)

        # We'd like to build blocks with alternating shifted windows
        assert (
            num_blocks % 2 == 0
        ), "The number of blocks must be even to use shifted windows"
        transform_pipeline = OrderedDict()
        for index in range(num_blocks):
            is_odd = index % 2 == 1
            transform_pipeline[f"block_{index}"] = SwinTransformerBlock(
                embedding_dim=embedding_dim,
                patch_dim=patch_dim,
                apply_shift=is_odd,
                image_shape=image_shape,
            )

        self.transform = nn.Sequential(transform_pipeline)

    def forward(self, patches: torch.Tensor):
        embeddings = self.preprocess(patches)
        transformed = self.transform(embeddings)

        return transformed


class BrazenNet(nn.Module):
    """A perception stack consisting of a SWIN transformer stage and a linear layer"""

    def __init__(
        self,
        patch_dim: int = 8,
        image_shape: tuple = (2048, 320),
        embedding_dim: int = 128,
        block_stages: list = [2, 2, 8, 2],
        merge_reduce_factor: int = 2,
    ):
        super().__init__()
        # Map input image to 1d patch tokens
        self.patch_partition = einops_torch.Rearrange(
            "batch 1 (vertical_patches patch_height) (horizontal_patches patch_width) -> batch vertical_patches horizontal_patches (patch_height patch_width)",
            patch_height=patch_dim,
            patch_width=patch_dim,
        )

        transforms = OrderedDict()
        for index, num_blocks in enumerate(block_stages):
            block_embedding_dim = embedding_dim * (
                2**num_blocks
            )  # scale 1, 2, 4, 8...
            apply_merge = index > 0
            transforms[f"stage_{index}"] = SwinTransformerStage(
                block_embedding_dim,
                num_blocks=num_blocks,
                apply_merge=apply_merge,
                patch_dim=patch_dim,
                image_shape=image_shape,
                merge_reduce_factor=merge_reduce_factor,
            )

        self.transforms = nn.Sequential(transforms)

    def forward(self, images: torch.Tensor):
        patches = self.patch_partition(images)
        transformed = self.transforms(patches)

        return transformed
