from audioop import bias
import math
from tkinter.tix import WINDOW
from typing import OrderedDict
from black import out

from einops.layers import torch as einops_torch
import einops
import torch
from torch import nn
import numpy as np

import utils
import dataset

# Following (horizontal, vertical) coordinates
WINDOW_PATCH_SHAPE = (8, 8)
PATCH_DIM = 8
EMBEDDING_DIM = 128  # roughly we want to increase dimensionality by the patch content for embeddings.
NUM_HEADS = 8
FEED_FORWARD_EXPANSION = 4  # Expansion factor for self attention feed-forward
BLOCK_STAGES = (2, 2, 2, 2, 2)  # Number of transformer blocks in each of the 4 stages
REDUCE_FACTOR = 2  # reduce factor (increase in patch size) in patch merging layer per stage


class SwinSelfAttention(nn.Module):
    """Implements locality self attention from https://arxiv.org/pdf/2112.13492.pdf"""

    _bias_index_pairs = torch.tensor(
        [[i, j] for i in range(WINDOW_PATCH_SHAPE[0]) for j in range(WINDOW_PATCH_SHAPE[1])],
        requires_grad=False,
    )
    _mask_view_indices = _bias_index_pairs[:, None, :] - _bias_index_pairs[None, :, :] + WINDOW_PATCH_SHAPE[0] - 1
    _position_bias_indices = _mask_view_indices[:, :, 0], _mask_view_indices[:, :, 1]

    @property
    def position_bias_indices(self):
        """Static class method to get the position bias indices.
        We use the same window shape for all layers, so no need to re-calculate.
        The position bias itself is a tensor of size (2 * patch_shape[1] - 1, 2 * patch_shape[0] - 1)
        We want to view it as a relative matrix in the full attention dimension
        """
        return type(self)._position_bias_indices

    def __init__(
        self,
        embedding_dim: int,
        window_shape: tuple,
        patch_shape: tuple = WINDOW_PATCH_SHAPE,
        num_heads: int = NUM_HEADS,
        apply_shift: tuple = None,
    ):
        super().__init__()

        self.head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0, "Dimension must be divisible by num_heads"

        # Learned embeddings for query, key and value
        self.attention_modes = ["query", "key", "value"]
        self.embeddings = {}
        for mode in self.attention_modes:
            self.embeddings[mode] = nn.Linear(embedding_dim, embedding_dim)

        # Attention mechanisms
        self.part_heads = einops_torch.Rearrange(
            "batch num_windows num_patches (num_heads embedding) -> batch num_heads num_windows num_patches embedding",
            num_heads=num_heads,
        )
        self.transpose_key = einops_torch.Rearrange(
            "batch num_heads num_windows num_patches embedding -> batch num_heads num_windows embedding num_patches",
        )

        # Learned position bias
        position_bias = nn.parameter.Parameter(torch.zeros((2 * patch_shape[1]) - 1, (2 * patch_shape[0]) - 1))
        self.position_bias = nn.init.trunc_normal_(position_bias, mean=0, std=0.02)

        # Mask for attention
        mask = torch.zeros(
            (
                window_shape[1],
                window_shape[0],
                patch_shape[1],
                patch_shape[0],
                patch_shape[1],
                patch_shape[0],
            ),
            dtype=torch.bool,
        )  # TODO: verify patch_shape[0] -> vertical, horizontal

        if apply_shift:
            shift_x, shift_y = (
                patch_shape[0] // 2,
                patch_shape[1] // 2,
            )

            # Prevent shifted patches from bottom from seeing the top as their neighbours
            mask[-1, :, :shift_y, :, shift_y:, :] = True
            mask[-1, :, shift_y:, :, :shift_y, :] = True

            # Prevent shifted patches from left from seeing the bottom as their neighbours
            mask[:, -1, :, :shift_x, :, shift_x:] = True
            mask[:, -1, :, shift_x:, :, :shift_x] = True

        self.mask = einops.rearrange(
            mask,
            utils.assemble_einops_string(
                input_shape="vertical_windows horizontal_windows vertical_patches_1 horizontal_patches_1 vertical_patches_2 horizontal_patches_2",
                output_shape="(vertical_windows horizontal_windows) (vertical_patches_1 horizontal_patches_1) (vertical_patches_2 horizontal_patches_2)"
            )
        )

        self.join_heads = einops_torch.Rearrange(
            "batch num_heads num_windows num_patches embedding -> batch num_windows num_patches (num_heads embedding)"
        )

    def forward(self, patches: torch.Tensor):
        heads = {}
        for mode in self.attention_modes:
            embedding_layer = self.embeddings[mode].to(patches.device)
            embedding = embedding_layer(patches)
            heads[mode] = self.part_heads(embedding)  # multi headed attention
        query, key_transposed, value = (
            heads["query"],
            self.transpose_key(heads["key"]),
            heads["value"],
        )

        attention_logits = torch.matmul(query, key_transposed) / math.sqrt(self.head_dim)
        biased_attention = (
            attention_logits + self.position_bias[self.position_bias_indices[0], self.position_bias_indices[1]]
        )

        # Mask relevant values
        mask = self.mask.to(patches.device)
        masked_attention = biased_attention.masked_fill(mask, float("-inf"))
        attention = nn.functional.softmax(masked_attention, dim=-1)

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
        image_shape: tuple = dataset.IMAGE_SHAPE,
        patch_shape: tuple = WINDOW_PATCH_SHAPE,
    ):
        super().__init__()
        for index, _ in enumerate(image_shape):
            assert (
                patch_shape[0] % 2 == 0 and patch_shape[1] % 2 == 0
            ), "Window shape must be even for even window splitting"
            assert image_shape[index] % patch_dim == 0, "Image width must be divisible by patch dimension"
            assert (image_shape[index] // patch_dim) % patch_shape[
                index
            ] == 0, "Number of patches must be divisible by window dimension"

        self.cyclic_shift = (
            (
                -(patch_shape[1] // 2),
                -(patch_shape[0] // 2),
            )  # switch order to vertical, horizontal to match embedding
            if apply_shift is True
            else None
        )
        self.metadata = {  # debugging metadata
            "embedding_dim": embedding_dim,
            "patch_dim": patch_dim,
            "patch_shape": patch_shape,
            "window_shape": (
                image_shape[0] // patch_dim // patch_shape[0],
                image_shape[1] // patch_dim // patch_shape[1],
            ),
        }

        self.part = einops_torch.Rearrange(
            utils.assemble_einops_string(
                input_shape="batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch",
                output_shape="batch (vertical_windows horizontal_windows) (vertical_patches horizontal_patches) patch"
            ),
            vertical_patches=patch_shape[1],
            horizontal_patches=patch_shape[0],
        )  # fix the number of patches per window, let it find number of windows from image
        self.join = einops_torch.Rearrange(
            utils.assemble_einops_string(
                input_shape="batch (vertical_windows horizontal_windows) (vertical_patches horizontal_patches) patch",
                output_shape="batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch"
            ),
            vertical_windows=self.metadata["window_shape"][1],
            vertical_patches=patch_shape[1],
        )

        attention = OrderedDict()
        attention["layer_norm"] = nn.LayerNorm(embedding_dim)
        attention["transform"] = SwinSelfAttention(
            embedding_dim,
            window_shape=self.metadata["window_shape"],
            patch_shape=patch_shape,
            apply_shift=apply_shift,
        )
        self.attention = nn.Sequential(attention)

        feed_forward = OrderedDict()
        feed_forward["layer_norm"] = nn.LayerNorm(embedding_dim)
        feed_forward["linear_1"] = nn.Linear(embedding_dim, embedding_dim * feed_forward_expansion)
        feed_forward["gelu"] = nn.GELU()
        feed_forward["linear_2"] = nn.Linear(embedding_dim * feed_forward_expansion, embedding_dim)
        self.feed_forward = nn.Sequential(feed_forward)

    def forward(self, embeddings: torch.Tensor):
        if self.cyclic_shift:
            embeddings = embeddings.roll(
                self.cyclic_shift, dims=(1, 2)
            )  # horizontal, vertical. TODO: explore einops.reduce

        patches = self.part(embeddings)
        attention = patches + self.attention(patches)
        output = attention + self.feed_forward(attention)
        output_patches = self.join(output)

        if self.cyclic_shift:
            output_patches = output_patches.roll((-self.cyclic_shift[0], -self.cyclic_shift[1]), dims=(1, 2))  # unroll

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
        image_shape: tuple = dataset.IMAGE_SHAPE,
        patch_shape: tuple = WINDOW_PATCH_SHAPE,
        merge_reduce_factor: int = REDUCE_FACTOR,
    ):
        super().__init__()

        input_pipeline = OrderedDict()
        if apply_merge:
            # Reduce patches - merge by the reduction factor to 1, widen the embeddings
            input_pipeline["reduce"] = einops_torch.Rearrange(
                utils.assemble_einops_string(
                    input_shape="batch (vertical_patches vertical_reduce) (horizontal_patches horizontal_reduce) patch",
                    output_shape="batch vertical_patches horizontal_patches (vertical_reduce horizontal_reduce patch)"
                ),
                vertical_reduce=merge_reduce_factor,
                horizontal_reduce=merge_reduce_factor,
            )

        input_pipeline["embed"] = nn.LazyLinear(embedding_dim)
        self.preprocess = nn.Sequential(input_pipeline)

        # We'd like to build blocks with alternating shifted windows
        assert num_blocks % 2 == 0, "The number of blocks must be even to use shifted windows"
        transform_pipeline = OrderedDict()
        for index in range(num_blocks):
            is_odd = index % 2 == 1
            transform_pipeline[f"block_{index}"] = SwinTransformerBlock(
                embedding_dim=embedding_dim,
                patch_dim=patch_dim,
                apply_shift=is_odd,
                image_shape=image_shape,
                patch_shape=patch_shape,
            )

        self.transform = nn.Sequential(transform_pipeline)

    def forward(self, patches: torch.Tensor):
        embeddings = self.preprocess(patches)
        transformed = self.transform(embeddings)

        return transformed


class BrazenNet(nn.Module):
    """A perception stack consisting of a SWIN transformer stage and a feed-forward layer
    Predicts the entire output sequence in one go
    """

    def __init__(
        self,
        patch_dim: int = PATCH_DIM,
        image_shape: tuple = dataset.IMAGE_SHAPE,
        patch_shape: tuple = WINDOW_PATCH_SHAPE,
        embedding_dim: int = EMBEDDING_DIM,
        block_stages: tuple = BLOCK_STAGES,
        merge_reduce_factor: int = REDUCE_FACTOR,
    ):
        super().__init__()
        encoder = OrderedDict()

        # Map greyscale input image to patch tokens
        encoder["extract_patches"] = einops_torch.Rearrange(
            utils.assemble_einops_string(
                input_shape="batch 1 (vertical_patches patch_height) (horizontal_patches patch_width)",
                output_shape="batch vertical_patches horizontal_patches (patch_height patch_width)"
            ),
            patch_height=patch_dim,
            patch_width=patch_dim,
        )

        # Apply visual self-attention
        patch_reduction_multiples = [merge_reduce_factor**index for index, _ in enumerate(block_stages)]
        for index, num_blocks in enumerate(block_stages):
            # Apply sequential swin transformer blocks, reducing the number of patches for each stage
            apply_merge = index > 0
            encoder[f"stage_{index}"] = SwinTransformerStage(
                embedding_dim * patch_reduction_multiples[index],
                num_blocks=num_blocks,
                apply_merge=apply_merge,
                patch_dim=patch_dim * patch_reduction_multiples[index],
                image_shape=image_shape,
                patch_shape=patch_shape,
                merge_reduce_factor=merge_reduce_factor,
            )

        # Rearrange window embeddings to a single embedding layer
        encoder["combine_outputs"] = einops_torch.Rearrange(
            "batch vertical_patches horizontal_patches embedding -> batch (vertical_patches horizontal_patches embedding)",
            vertical_patches=patch_shape[1],
            horizontal_patches=patch_shape[0],
        )

        self.encoder = nn.Sequential(encoder)

        # Map transformer outputs to sequence of symbols
        # TODO: Here will be a transformer decoder. In comes the masked output self attention along with the encoder output.
        feed_forward = OrderedDict()
        encoder_output_dim = patch_shape[0] * patch_shape[1] * (embedding_dim * patch_reduction_multiples[-1])
        feed_forward_expansion_dim = (dataset.SYMBOLS_DIM + 1) * FEED_FORWARD_EXPANSION
        feed_forward["linear_0"] = nn.Linear(encoder_output_dim, feed_forward_expansion_dim)
        feed_forward["gelu"] = nn.GELU()

        output_dim = (dataset.SYMBOLS_DIM + 1) * dataset.SEQUENCE_DIM
        feed_forward["linear_1"] = nn.Linear(feed_forward_expansion_dim, output_dim)
        feed_forward["reshape"] = einops_torch.Rearrange(
            "batch (output_sequence output_symbol) -> batch output_sequence output_symbol",
            output_sequence=dataset.SEQUENCE_DIM,
            output_symbol=(dataset.SYMBOLS_DIM + 1),
        )
        feed_forward["softmax"] = nn.Softmax(dim=-1)

        self.feed_forward = nn.Sequential(feed_forward)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, images: torch.Tensor):
        transformed = self.encoder(images)
        assert transformed.shape[1:3] == (
            WINDOW_PATCH_SHAPE[1],
            WINDOW_PATCH_SHAPE[0],
        ), "The output was not reduced to a single window at the final output stage"
        global_embeddings = self.combine_outputs(transformed)
        output_sequence = self.output(global_embeddings)

        return output_sequence
