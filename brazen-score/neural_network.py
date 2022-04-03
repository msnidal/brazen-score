import math
from typing import OrderedDict

from einops.layers import torch as einops_torch
import einops
import torch
from torch import nn

import dataset

# Following (horizontal, vertical) coordinates
WINDOW_PATCH_SHAPE = (16, 10)
PATCH_DIM = 16
EMBEDDING_DIM = 64  # roughly we want to increase dimensionality by the patch content for embeddings. Experimenting with lower value
NUM_HEADS = 8
FEED_FORWARD_EXPANSION = 4  # Expansion factor for self attention feed-forward
BLOCK_STAGES = (2, 2, 8, 2)  # Number of transformer blocks in each of the 4 stages
REDUCE_FACTOR = (
    2  # reduce factor (increase in patch size) in patch merging layer per stage
)


class SwinTransformer(nn.Module):
    """Implements locality self attention from https://arxiv.org/pdf/2112.13492.pdf
    Attention(Q, K, V) = SoftMax(QK^T / sqrt(d) + B)V
    """

    def __init__(
        self,
        embedding_dim: int,
        image_window_shape: tuple,
        window_patch_shape: tuple = WINDOW_PATCH_SHAPE,
        num_heads: int = NUM_HEADS,
        apply_shift: tuple = None,
    ):
        super().__init__()

        self.head_dim = embedding_dim // num_heads
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
            "batch num_windows num_patches (num_heads embedding) -> batch num_heads num_windows num_patches embedding",
            num_heads=num_heads,
        )
        self.transpose_key = einops_torch.Rearrange(
            "batch num_heads num_windows num_patches embedding -> batch num_heads num_windows embedding num_patches",
        )

        # Learned position bias
        position_bias = nn.parameter.Parameter(
            torch.Tensor(
                size=((2 * window_patch_shape[1]) - 1, (2 * window_patch_shape[0]) - 1)
            )
        )
        self.position_bias = nn.init.trunc_normal_(position_bias, mean=0, std=0.02)
        self.window_patch_shape = window_patch_shape
        # we need to project this to position_bias_dim

        attention_mask = torch.zeros(
            (
                image_window_shape[1],
                image_window_shape[0],
                window_patch_shape[1],
                window_patch_shape[0],
                window_patch_shape[1],
                window_patch_shape[0],
            ),
            dtype=torch.bool,
        )  # TODO: verify window_patch_shape[0] -> vertical, horizontal

        if apply_shift:
            horizontal_patch_displacement, vertical_patch_displacement = (
                window_patch_shape[0] // 2,
                window_patch_shape[1] // 2,
            )

            # Prevent shifted patches from bottom from seeing the top as their neighbours
            attention_mask[
                -1, :, :vertical_patch_displacement, :, vertical_patch_displacement:, :
            ] = True
            attention_mask[
                -1, :, vertical_patch_displacement:, :, :vertical_patch_displacement, :
            ] = True

            # Prevent shifted patches from left from seeing the bottom as their neighbours
            attention_mask[
                :,
                -1,
                :,
                :horizontal_patch_displacement,
                :,
                horizontal_patch_displacement:,
            ] = True
            attention_mask[
                :,
                -1,
                :,
                horizontal_patch_displacement:,
                :,
                :horizontal_patch_displacement,
            ] = True

        self.attention_mask = einops.rearrange(
            attention_mask,
            "vertical_windows horizontal_windows vertical_patches_1 horizontal_patches_1 vertical_patches_2 horizontal_patches_2 -> (vertical_windows horizontal_windows) (vertical_patches_1 horizontal_patches_1) (vertical_patches_2 horizontal_patches_2)",
        )

        self.join_heads = einops_torch.Rearrange(
            "batch num_heads num_windows num_patches embedding -> batch num_windows num_patches (num_heads embedding)"
        )

    def relative_position_bias(self, attention_logits):
        expanded_logits = einops.rearrange(
            attention_logits,
            "batch num_heads num_windows (vertical_patches_base horizontal_patches_base) (vertical_patches_target horizontal_patches_target) -> batch num_heads num_windows vertical_patches_base horizontal_patches_base vertical_patches_target horizontal_patches_target",
            vertical_patches_base=self.window_patch_shape[1],
            horizontal_patches_base=self.window_patch_shape[0],
            vertical_patches_target=self.window_patch_shape[1],
            horizontal_patches_target=self.window_patch_shape[0],
        )

        for vertical_base in range(self.window_patch_shape[1]):
            for horizontal_base in range(self.window_patch_shape[0]):
                for vertical_target in range(self.window_patch_shape[1]):
                    for horizontal_target in range(self.window_patch_shape[0]):
                        relative_vertical_position = vertical_target - vertical_base
                        relative_horizontal_position = (
                            horizontal_target - horizontal_base
                        )

                        expanded_logits[
                            :,
                            :,
                            :,
                            vertical_base,
                            horizontal_base,
                            vertical_target,
                            horizontal_target,
                        ] += self.position_bias[
                            relative_vertical_position, relative_horizontal_position
                        ]

        return einops.rearrange(
            expanded_logits,
            "batch num_heads num_windows vertical_base horizontal_base vertical_target horizontal_target -> batch num_heads num_windows (vertical_base horizontal_base) (vertical_target horizontal_target)",
            vertical_base=self.window_patch_shape[1],
            horizontal_base=self.window_patch_shape[0],
            vertical_target=self.window_patch_shape[1],
            horizontal_target=self.window_patch_shape[0],
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

        attention_logits = torch.matmul(query, key_transposed) / math.sqrt(
            self.head_dim
        )
        biased_attention = self.relative_position_bias(attention_logits)

        # Mask relevant values
        attention_mask = self.attention_mask.to(patches.device)
        masked_attention = biased_attention.masked_fill(attention_mask, float("-inf"))
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
        window_patch_shape: tuple = WINDOW_PATCH_SHAPE,
    ):
        super().__init__()
        for index, _ in enumerate(image_shape):
            assert (
                window_patch_shape[0] % 2 == 0 and window_patch_shape[1] % 2 == 0
            ), "Window shape must be even for even window splitting"
            assert (
                image_shape[index] % patch_dim == 0
            ), "Image width must be divisible by patch dimension"
            assert (image_shape[index] // patch_dim) % window_patch_shape[
                index
            ] == 0, "Number of patches must be divisible by window dimension"

        self.cyclic_shift = (
            (
                -(window_patch_shape[1] // 2),
                -(window_patch_shape[0] // 2),
            )  # switch order to vertical, horizontal to match embedding
            if apply_shift is True
            else None
        )
        self.metadata = {  # debugging metadata
            "embedding_dim": embedding_dim,
            "patch_dim": patch_dim,
            "window_patch_shape": window_patch_shape,
            "image_window_shape": (
                image_shape[0] // patch_dim // window_patch_shape[0],
                image_shape[1] // patch_dim // window_patch_shape[1],
            ),
        }

        self.part = einops_torch.Rearrange(
            "batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch -> batch (vertical_windows horizontal_windows) (vertical_patches horizontal_patches) patch",
            vertical_patches=window_patch_shape[1],
            horizontal_patches=window_patch_shape[0],
        )  # fix the number of patches per window, let it find number of windows from image
        self.join = einops_torch.Rearrange(
            "batch (vertical_windows horizontal_windows) (vertical_patches horizontal_patches) patch -> batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch",
            vertical_windows=self.metadata["image_window_shape"][1],
            vertical_patches=window_patch_shape[1],
        )

        attention = OrderedDict()
        attention["layer_norm"] = nn.LayerNorm(embedding_dim)
        attention["transform"] = SwinTransformer(
            embedding_dim,
            image_window_shape=self.metadata["image_window_shape"],
            window_patch_shape=window_patch_shape,
            apply_shift=apply_shift,
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
        if self.cyclic_shift:
            embeddings = embeddings.roll(
                self.cyclic_shift, dims=(1, 2)
            )  # horizontal, vertical. TODO: explore einops.reduce

        patches = self.part(embeddings)
        attention = patches + self.attention(patches)
        output = attention + self.feed_forward(attention)
        output_patches = self.join(output)

        if self.cyclic_shift:
            output_patches = output_patches.roll(
                (-self.cyclic_shift[0], -self.cyclic_shift[1]), dims=(1, 2)
            )  # unroll

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
        window_patch_shape: tuple = WINDOW_PATCH_SHAPE,
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
                window_patch_shape=window_patch_shape,
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
        window_patch_shape: tuple = WINDOW_PATCH_SHAPE,
        embedding_dim: int = EMBEDDING_DIM,
        block_stages: tuple = BLOCK_STAGES,
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
        stage_multipliers = []
        for index, num_blocks in enumerate(block_stages):
            stage_multipliers.append(merge_reduce_factor**index)
            apply_merge = index > 0
            transforms[f"stage_{index}"] = SwinTransformerStage(
                embedding_dim * stage_multipliers[index],
                num_blocks=num_blocks,
                apply_merge=apply_merge,
                patch_dim=patch_dim * stage_multipliers[index],
                image_shape=image_shape,
                window_patch_shape=window_patch_shape,
                merge_reduce_factor=merge_reduce_factor,
            )

        self.transforms = nn.Sequential(transforms)

        self.combine_outputs = einops_torch.Rearrange(
            "batch vertical_patches horizontal_patches patch -> batch (vertical_patches horizontal_patches patch)",
            vertical_patches=window_patch_shape[1],
            horizontal_patches=window_patch_shape[0],
        )

        # Map transformer outputs to sequence of symbols
        output = OrderedDict()
        output_embedding_dim = (
            window_patch_shape[0]
            * window_patch_shape[1]
            * (embedding_dim * stage_multipliers[-1])
        )  # Linearize
        output_dim = (dataset.SYMBOLS_DIM + 1) * dataset.SEQUENCE_DIM
        inner_output_dim = (dataset.SYMBOLS_DIM + 1) * FEED_FORWARD_EXPANSION
        output["linear_0"] = nn.Linear(output_embedding_dim, inner_output_dim)
        output["gelu"] = nn.GELU()
        output["linear_1"] = nn.Linear(inner_output_dim, output_dim)
        output["reshape"] = einops_torch.Rearrange(
            "batch (output_sequence output_symbol) -> batch output_sequence output_symbol",
            output_sequence=dataset.SEQUENCE_DIM,
            output_symbol=(dataset.SYMBOLS_DIM + 1),
        )
        output["softmax"] = nn.Softmax(dim=-1)

        self.output = nn.Sequential(output)

    def forward(self, images: torch.Tensor):
        patches = self.patch_part(images)
        transformed = self.transforms(patches)
        assert transformed.shape[1:3] == (
            WINDOW_PATCH_SHAPE[1],
            WINDOW_PATCH_SHAPE[0],
        ), "The output was not reduced to a single window at the final output stage"
        global_embeddings = self.combine_outputs(transformed)
        output_sequence = self.output(global_embeddings)

        return output_sequence
