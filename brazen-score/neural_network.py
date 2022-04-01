import math
from typing import OrderedDict

from einops.layers import torch as einops_torch
import torch
from torch import nn

import dataset

WINDOW_SHAPE = (8, 6)
PATCH_DIM = 8
EMBEDDING_DIM = (
    PATCH_DIM**2
) * 2  # roughly we want to increase dimensionality by the patch content for embeddings
NUM_HEADS = 8
FEED_FORWARD_EXPANSION = 4  # ff expansion factor in self attention
BLOCK_STAGES = (2, 2, 8, 2)
REDUCE_FACTOR = 2  # reduce factor in patch merging layer per stage


class SwinTransformer(nn.Module):
    """Implements locality self attention from https://arxiv.org/pdf/2112.13492.pdf
    Attention(Q, K, V) = SoftMax(QK^T / sqrt(d) + B)V
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = NUM_HEADS,
        window_shape: tuple[int] = WINDOW_SHAPE,
    ):
        super().__init__()

        self.dimension = embedding_dim
        assert (
            embedding_dim % num_heads == 0
        ), "Dimension must be divisible by num_heads"

        # Learned embeddings for query, key and value
        self.attention_modes = ["query", "key", "value"]
        self.embeddings = {}
        for mode in self.attention_modes:
            self.embeddings[mode] = nn.Linear(embedding_dim, embedding_dim)

        # Attention mechanisms
        self.part_heads = einops_torch.Rearrange(
            "batch num_windows num_patches (num_heads embedding) -> batch num_windows num_heads num_patches embedding",
            num_heads=num_heads,
        )
        self.transpose_key = einops_torch.Rearrange(
            "batch num_windows num_heads num_patches embedding -> batch num_windows num_heads embedding num_patches",
        )

        # Learned position bias
        position_bias_dim = window_shape[0] * window_shape[1]  # (window_dim * 2) - 1
        position_bias = nn.parameter.Parameter(
            torch.Tensor(size=(num_heads, position_bias_dim, position_bias_dim))
        )
        self.position_bias = nn.init.trunc_normal_(position_bias, mean=0, std=0.02)

        self.join_heads = einops_torch.Rearrange(
            "batch num_windows num_heads num_patches embedding -> batch num_windows num_patches (num_heads embedding)"
        )

    def forward(self, patches: torch.Tensor):
        heads = {}
        for mode in self.attention_modes:
            embedding = self.embeddings[mode](patches)
            heads[mode] = self.part_heads(embedding)  # multi headed attention
        query, key_transposed, value = (
            heads["query"],
            self.transpose_key(heads["key"]),
            heads["value"],
        )

        attention_logits = (
            torch.matmul(query, key_transposed) / math.sqrt(self.dimension)
        ) + self.position_bias  # + self.mask
        attention = nn.functional.softmax(attention_logits, dim=-1)

        self_attention = torch.matmul(attention, value)
        output = self.join_heads(self_attention)

        return output


class SwinTransformerBlock(nn.Module):
    """Applies a single layer of the transformer to the input"""

    def __init__(
        self,
        embedding_dim: int,
        patch_dim: int,
        apply_shift: bool = False,
        feed_forward_expansion: int = FEED_FORWARD_EXPANSION,
        image_shape: tuple[int] = dataset.IMAGE_SHAPE,
        window_shape: tuple[int] = WINDOW_SHAPE,
    ):
        super().__init__()
        for index, _ in enumerate(image_shape):
            assert (
                window_shape[0] % 2 == 0 and window_shape[1] % 2 == 0
            ), "Window shape must be even for even window splitting"
            assert (
                image_shape[index] % patch_dim == 0
            ), "Image width must be divisible by patch dimension"
            assert (image_shape[index] // patch_dim) % window_shape[
                index
            ] == 0, "Number of patches must be divisible by window dimension"

        self.apply_shift = (
            (window_shape[0] // 2, window_shape[1] // 2)
            if apply_shift is True
            else None
        )

        self.part = einops_torch.Rearrange(
            "batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch -> batch (vertical_windows horizontal_windows) (vertical_patches horizontal_patches) patch",
            vertical_patches=window_shape[1],
            horizontal_patches=window_shape[0],
        )  # fix the number of patches per window, let it find number of windows from image
        self.join = einops_torch.Rearrange(
            "batch (vertical_windows horizontal_windows) (vertical_patches horizontal_patches) patch -> batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch",
            vertical_windows=image_shape[1] // patch_dim // window_shape[1],
            vertical_patches=window_shape[1],
        )

        attention = OrderedDict()
        attention["layer_norm"] = nn.LayerNorm(embedding_dim)
        attention["transform"] = SwinTransformer(
            embedding_dim, window_shape=window_shape
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
            )  # horizontal, vertical. TODO: explore einops.reduce

        patches = self.part(embeddings)
        attention = patches + self.attention(patches)
        output = attention + self.feed_forward(attention)
        output_patches = self.join(output)

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
        patch_dim: int = PATCH_DIM,
        image_shape: tuple[int] = dataset.IMAGE_SHAPE,
        window_shape: tuple[int] = WINDOW_SHAPE,
        merge_reduce_factor: int = REDUCE_FACTOR,
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
                window_shape=window_shape,
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
        patch_dim: int = PATCH_DIM,
        image_shape: tuple[int] = dataset.IMAGE_SHAPE,
        window_shape: tuple[int] = WINDOW_SHAPE,
        embedding_dim: int = EMBEDDING_DIM,
        block_stages: tuple[int] = BLOCK_STAGES,
        merge_reduce_factor: int = REDUCE_FACTOR,
    ):
        super().__init__()
        # Map input image to 1d patch tokens
        self.patch_part = einops_torch.Rearrange(
            "batch 1 (vertical_patches patch_height) (horizontal_patches patch_width) -> batch vertical_patches horizontal_patches (patch_height patch_width)",
            patch_height=patch_dim,
            patch_width=patch_dim,
        )

        # Apply visual self attention
        transforms = OrderedDict()
        for index, num_blocks in enumerate(block_stages):
            block_embedding_dim = embedding_dim * (2**index)  # scale 1, 2, 4, 8...
            apply_merge = index > 0
            transforms[f"stage_{index}"] = SwinTransformerStage(
                block_embedding_dim,
                num_blocks=num_blocks,
                apply_merge=apply_merge,
                patch_dim=patch_dim,
                image_shape=image_shape,
                window_shape=window_shape,
                merge_reduce_factor=merge_reduce_factor,
            )

        self.transforms = nn.Sequential(transforms)

        # Map to output sequence. Explore use of transformer or direct output
        self.output = (75, 758)
        # self.output_part = einops_torch.Rearrange(
        #    "batch vertical_patches horizontal_patches (patch_height patch_width) -> batch (vertical_patches horizontal_patches) patch_height patch_width",
        # )

    def forward(self, images: torch.Tensor):
        patches = self.patch_part(images)
        transformed = self.transforms(patches)
        # literally output a 75 x 768 softmaxx tensor LMAO
        # output = self.output_part(transformed)

        return output
