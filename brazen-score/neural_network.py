from audioop import bias
import math
from tkinter.tix import WINDOW
from turtle import position
from typing import OrderedDict
from black import out

from einops.layers import torch as einops_torch
import einops
import torch
from torch import nn
from torch.nn import functional
import numpy as np

import utils
import dataset


# Following (horizontal, vertical) coordinates
WINDOW_PATCH_SHAPE = (8, 8)
PATCH_DIM = 8
ENCODER_EMBEDDING_DIM = 64
DECODER_EMBEDDING_DIM = 1024
NUM_HEADS = 4
FEED_FORWARD_EXPANSION = 2  # Expansion factor for self attention feed-forward
ENCODER_BLOCK_STAGES = (2, 2)  # Number of transformer blocks in each of the 4 stages
NUM_DECODER_BLOCKS = 1  # Number of decoder blocks
REDUCE_FACTOR = 16  # reduce factor (increase in patch size) in patch merging layer per stage


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -1.0, 1.0)
    elif isinstance(module, nn.parameter.Parameter):
        nn.init.trunc_normal_(module, 0.0, std=0.02)


class BrazenParameters:
    def __init__(
        self,
        window_patch_shape=WINDOW_PATCH_SHAPE,
        image_shape=dataset.IMAGE_SHAPE,
        patch_dim=PATCH_DIM,
        encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
        decoder_embedding_dim=DECODER_EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        feed_forward_expansion=FEED_FORWARD_EXPANSION,
        encoder_block_stages=ENCODER_BLOCK_STAGES,
        num_decoder_blocks=NUM_DECODER_BLOCKS,
        reduce_factor=REDUCE_FACTOR,
        output_sequence_dim=dataset.SEQUENCE_DIM,
        num_symbols=dataset.NUM_SYMBOLS,
    ):
        params = locals()
        for param in params:
            setattr(self, param, params[param])


class MultiHeadAttention(nn.Module):
    """Generic multi-head attention module implemented throughout both visual and output sequence transformer"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mask: torch.Tensor = None,
        position_bias_dim: int = None,
        position_bias_indices: tuple = None,
        shape_prefix="batch",
    ):
        super().__init__()

        # self.reshape_input = reshape_input # TODO: pass in prefix string
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.split_heads = einops_torch.Rearrange(
            f"{shape_prefix} sequence (num_heads head_embedding) -> {shape_prefix} num_heads sequence head_embedding",
            num_heads=num_heads,
            head_embedding=self.head_dim,
        )
        self.join_heads = einops_torch.Rearrange(
            f"{shape_prefix} num_heads sequence head_embedding -> {shape_prefix} sequence (num_heads head_embedding)",
            num_heads=num_heads,
            head_embedding=self.head_dim,
        )

        # Learned embeddings for query, key and value
        for name in ["query", "key", "value"]:
            modules = OrderedDict()
            for index in range(num_heads):
                modules[str(index)] = nn.Linear(embedding_dim, self.head_dim)

            setattr(self, f"embed_{name}", nn.ModuleDict(modules))

        # self.embed_query, self.embed_key, self.embed_value = nn.Linear(embedding_dim, embedding_dim), nn.Linear(embedding_dim, embedding_dim), nn.Linear(embedding_dim, embedding_dim)
        self.output_embedding = nn.Linear(embedding_dim, embedding_dim)

        # Optional properties
        # assert mask.shape == (self. self.head_dim, self.head_dim), f"Mask is {mask.shape}, must be square matrix of shape {self.head_dim}x{self.head_dim} (Embedding dim {embedding_dim} // Num heads {num_heads})"
        if mask is not None:
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

        if position_bias_dim is not None:
            assert position_bias_indices is not None, "Position bias indices must be specified"
            self.position_bias_indices = position_bias_indices

            # Learned position bias
            self.position_bias = nn.parameter.Parameter(torch.zeros(position_bias_dim))
        else:
            self.position_bias = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """Apply multi-head attention to query, key and value"""

        # Apply embeddings and split heads
        attention_heads = []
        for index in range(self.num_heads):
            query_head = self.embed_query[str(index)](query)
            key_head = self.embed_key[str(index)](key)
            value_head = self.embed_value[str(index)](value)

            attention_logits = torch.div(torch.matmul(query_head, key_head.transpose(-1, -2)), math.sqrt(self.head_dim))

            # Optional position bias for swin transforms
            if self.position_bias is not None:
                attention_logits = (
                    attention_logits + self.position_bias[self.position_bias_indices[0], self.position_bias_indices[1]]
                )

            # Mask relevant values, apply softmax
            if self.mask is not None:
                attention_logits = attention_logits.masked_fill(self.mask, float("-inf"))
            attention = self.softmax(attention_logits)

            # Apply attention
            output = torch.matmul(attention, value_head)
            attention_heads.append(output)

        concatenated_heads = torch.cat(attention_heads, dim=-1)
        output_embedding = self.output_embedding(concatenated_heads)

        return output_embedding


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

    def generate_mask(self, window_shape: int, patch_shape: int, num_heads: int, apply_shift: tuple = None):
        """Create the unique self-attention mask, taking into consideration window shifting"""

        mask = torch.zeros(
            (
                window_shape[1],
                window_shape[0],
                patch_shape[1],
                patch_shape[0],
                patch_shape[1],
                patch_shape[0],
            ),
            names=("window_y", "window_x", "base_patch_y", "base_patch_x", "target_patch_y", "target_patch_x"),
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

        arranged_mask = einops.rearrange(
            mask.rename(None),
            utils.assemble_einops_string(
                input_shape="vertical_windows horizontal_windows vertical_patches_base horizontal_patches_base vertical_patches_target horizontal_patches_target",
                output_shape="vertical_windows horizontal_windows (vertical_patches_base horizontal_patches_base) (vertical_patches_target horizontal_patches_target)",
            ),
        )
        return arranged_mask

    def __init__(
        self,
        embedding_dim: int,
        window_shape: tuple,
        patch_shape: tuple = WINDOW_PATCH_SHAPE,
        num_heads: int = NUM_HEADS,
        apply_shift: tuple = None,
    ):
        super().__init__()

        position_bias_dim = (2 * patch_shape[1] - 1, 2 * patch_shape[0] - 1)
        mask = self.generate_mask(window_shape, patch_shape, num_heads, apply_shift=apply_shift)
        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            mask=mask,
            position_bias_dim=position_bias_dim,
            position_bias_indices=self.position_bias_indices,
            shape_prefix="batch vertical_windows horizontal_windows",
        )

    def forward(self, patches: torch.Tensor):
        output = self.attention(patches, patches, patches)

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

        self.partition_windows = einops_torch.Rearrange(
            utils.assemble_einops_string(
                input_shape="batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch",
                output_shape="batch vertical_windows horizontal_windows (vertical_patches horizontal_patches) patch",
            ),
            vertical_patches=patch_shape[1],
            horizontal_patches=patch_shape[0],
        )  # fix the number of patches per window, let it find number of windows from image
        self.join_windows = einops_torch.Rearrange(
            utils.assemble_einops_string(
                input_shape="batch vertical_windows horizontal_windows (vertical_patches horizontal_patches) patch",
                output_shape="batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch",
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
            embeddings = embeddings.roll(self.cyclic_shift, dims=(1, 2))  # base[:][0][0]  == shifted[:][4][4]

        patches = self.partition_windows(embeddings)
        attention = patches + self.attention(patches)
        output = attention + self.feed_forward(attention)
        output_patches = self.join_windows(output)

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
        input_dim: int,
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
                    output_shape="batch vertical_patches horizontal_patches (vertical_reduce horizontal_reduce patch)",
                ),
                vertical_reduce=merge_reduce_factor,
                horizontal_reduce=merge_reduce_factor,
            )

        input_pipeline["embed"] = nn.Linear(input_dim, embedding_dim)
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


class DecoderBlock(nn.Module):
    """Decode all of the tokens in the output sequence as well as the Swin self-attention output"""

    def __init__(self, output_length: int, embedding_dim: int, feed_forward_expansion: int = FEED_FORWARD_EXPANSION):
        super().__init__()

        self.output_length = output_length

        attention_mask = torch.tensor(np.triu(np.full((output_length, output_length), True), 1).astype(np.bool))
        self.output_attention = MultiHeadAttention(embedding_dim, 8, attention_mask, shape_prefix="batch")
        self.output_attention_norm = nn.LayerNorm(embedding_dim)

        self.attention_decoder = MultiHeadAttention(embedding_dim, 8, shape_prefix="batch")
        self.attention_decoder_norm = nn.LayerNorm(embedding_dim)

        feed_forward = OrderedDict()
        feed_forward["linear_0"] = nn.Linear(embedding_dim, embedding_dim * feed_forward_expansion)
        feed_forward["gelu"] = nn.GELU()
        feed_forward["linear_1"] = nn.Linear(embedding_dim * feed_forward_expansion, embedding_dim)
        self.feed_forward = nn.Sequential(feed_forward)

        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings: dict):
        # output_sequence = torch.tensor([float("-inf") for _ in range(self.output_length)]) # Begin masked
        output_attention = self.output_attention_norm(
            embeddings["decoder"]
            + self.output_attention(embeddings["decoder"], embeddings["decoder"], embeddings["decoder"])
        )
        attention_decoder_outputs = self.attention_decoder_norm(
            output_attention + self.attention_decoder(embeddings["encoder"], embeddings["encoder"], output_attention)
        )
        feed_forward = self.feed_forward_norm(attention_decoder_outputs + self.feed_forward(attention_decoder_outputs))

        return {"decoder": feed_forward, "encoder": embeddings["encoder"]}


class BrazenNet(nn.Module):
    """A perception stack consisting of a SWIN transformer stage and a feed-forward layer
    Predicts the entire output sequence in one go
    """

    def __init__(
        self,
        config: BrazenParameters,
    ):
        super().__init__()
        self.config = config

        # Map greyscale input image to patch tokens
        self.extract_encoder_patches = einops_torch.Rearrange(
            utils.assemble_einops_string(
                input_shape="batch (vertical_patches patch_height) (horizontal_patches patch_width)",
                output_shape="batch vertical_patches horizontal_patches (patch_height patch_width)",
            ),
            patch_height=config.patch_dim,
            patch_width=config.patch_dim,
        )

        encoder = OrderedDict()
        # Apply visual self-attention
        patch_reduction_multiples = [
            config.reduce_factor**index for index, _ in enumerate(config.encoder_block_stages)
        ]
        input_dim = [config.patch_dim**2] + [
            config.encoder_embedding_dim * patch_reduction_multiples[i + 1] * config.reduce_factor
            for i in range(len(config.encoder_block_stages) - 1)
        ]
        for index, num_blocks in enumerate(config.encoder_block_stages):
            # Apply sequential swin transformer blocks, reducing the number of patches for each stage
            apply_merge = index > 0
            encoder[f"stage_{index}"] = SwinTransformerStage(
                config.encoder_embedding_dim * patch_reduction_multiples[index],
                input_dim[index],
                num_blocks=num_blocks,
                apply_merge=apply_merge,
                patch_dim=config.patch_dim * patch_reduction_multiples[index],
                image_shape=config.image_shape,
                patch_shape=config.window_patch_shape,
                merge_reduce_factor=config.reduce_factor,
            )

        self.encoder = nn.Sequential(encoder)

        # Rearrange window embeddings to a single embedding layer
        encoder_embeddings = OrderedDict()
        encoder_embeddings["combine_outputs"] = einops_torch.Rearrange(
            "batch vertical_patches horizontal_patches embedding -> batch (vertical_patches horizontal_patches embedding)",
            vertical_patches=config.window_patch_shape[1],
            horizontal_patches=config.window_patch_shape[0],
        )
        encoder_out_patch_dim = patch_reduction_multiples[-1] * config.encoder_embedding_dim
        encoder_out_patches = (
            config.window_patch_shape[0] * config.window_patch_shape[1]
        )  # assumes reductino to single window, is tested

        encoder_embeddings["reduce"] = nn.Linear(
            encoder_out_patches * encoder_out_patch_dim, config.decoder_embedding_dim
        )
        self.embed_encoder_output = nn.Sequential(encoder_embeddings)

        self.output_length = config.output_sequence_dim
        self.num_symbols = config.num_symbols

        self.embed_label = nn.Embedding(
            config.num_symbols + 1, config.decoder_embedding_dim, padding_idx=config.num_symbols
        )

        decoder = OrderedDict()
        for index in range(config.num_decoder_blocks):
            decoder["block_{}".format(index)] = DecoderBlock(config.output_sequence_dim, config.decoder_embedding_dim)
        self.decoder = nn.Sequential(decoder)

        # Map transformer outputs to sequence of symbols
        output = OrderedDict()
        output["linear_out"] = nn.Linear(config.decoder_embedding_dim, config.num_symbols + 1)
        #output["softmax"] = nn.LogSoftmax(dim=-1)

        self.output = nn.Sequential(output)

    def forward(self, images: torch.Tensor, labels: torch.Tensor = None):
        encoder_patches = self.extract_encoder_patches(images)
        encoded_images = self.encoder(encoder_patches)

        assert encoded_images.shape[1:3] == (
            self.config.window_patch_shape[1],
            self.config.window_patch_shape[0],
        ), "The output was not reduced to a single window at the final output stage. Check window shape, reduce factor, and encoder block stages."

        encoder_embeddings = self.embed_encoder_output(encoded_images)

        embeddings = {
            "encoder": einops.repeat(
                encoder_embeddings,
                "batch embedding -> batch sequence_length embedding",
                sequence_length=self.output_length,
            )
        }

        if labels is None:  # the model is being used in inference mdoe
            labels = torch.tensor(
                [self.num_symbols for _ in range(self.output_length)], device=images.device
            )  # Begin masked
            batch_labels = einops.repeat(labels, "symbol -> batch symbol", batch=2)
            embeddings["decoder"] = self.embed_label(batch_labels)

            for index in range(self.output_length):
                decoder_outputs = self.decoder(embeddings)
                output_sequence = self.output(decoder_outputs["decoder"])
                labels[: index + 1] = torch.max(output_sequence, dim=-1)[1][0, : index + 1]

            loss = None
        else:
            embeddings["decoder"] = self.embed_label(labels)

            decoder_outputs = self.decoder(embeddings)
            output_sequence = self.output(decoder_outputs["decoder"])
            loss = functional.cross_entropy(output_sequence, labels, reduction="mean", ignore_index=self.num_symbols)
            #loss = functional.nll_loss(output_sequence.transpose(2, 1), labels, reduction="sum")

        return output_sequence, loss
