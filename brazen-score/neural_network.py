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
import train
import parameters

KEY = "key"
QUERY = "query"
VALUE = "value"
ENCODER = "encoder"
DECODER = "decoder"


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


class MultiHeadAttention(nn.Module):
    """ Generic multi-head attention module implemented throughout both visual and output sequence transformer """

    def __init__(
        self,
        embedding_dim: int,
        config: parameters.BrazenParameters,
        mask: torch.Tensor = None,
        position_bias_dim: int = None,
        position_bias_indices: tuple = None,
    ):
        super().__init__()

        assert embedding_dim % config.num_heads == 0, "Embedding dim must be divisible by number of heads"
        self.num_heads = config.num_heads
        self.head_dim = embedding_dim // config.num_heads

        # Learned linear projections for query, key and value for each head
        self.project_heads = nn.ModuleDict(
            {mode: nn.ModuleList([nn.Linear(embedding_dim, self.head_dim) for _ in range(config.num_heads)]) for mode in [KEY, QUERY, VALUE]}
        )

        # Optional properties
        if mask is not None:
            #assert mask.shape == (self. self.head_dim, self.head_dim), f"Mask is {mask.shape}, must be square matrix of shape {self.head_dim}x{self.head_dim} (Embedding dim {embedding_dim} // Num heads {config.num_heads})"
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
        self.attention_dropout = nn.Dropout(config.dropout_rate)

        self.embed_output = nn.Linear(embedding_dim, embedding_dim)
        self.output_dropout = nn.Dropout(config.dropout_rate)

    def forward(self, embeddings: dict): # query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """Apply multi-head attention to query, key and value
        Embeddings should have three keys: "query", "key" and "value"
        """

        assert QUERY in embeddings and KEY in embeddings and VALUE in embeddings, "The multihead attention module takes a dictionary input"
        representation_heads = []

        for head_index in range(self.num_heads):
            head_embeddings = {mode: self.project_heads[mode][head_index](embeddings[mode]) for mode in self.project_heads.keys()}

            attention_scores = (head_embeddings[QUERY] @ head_embeddings[KEY].transpose(-1, -2)) / math.sqrt(self.head_dim)

            # Optional position bias for swin transforms
            if self.position_bias is not None:
                attention_scores = (
                    attention_scores + self.position_bias[self.position_bias_indices[0], self.position_bias_indices[1]]
                )
            if self.mask is not None:
                attention_scores = attention_scores.masked_fill(self.mask, float("-inf"))

            attention_weights = self.softmax(attention_scores)
            attention_weights = self.attention_dropout(attention_weights)

            # Apply attention
            representation = attention_weights @ head_embeddings[VALUE]
            representation_heads.append(representation)

        representations = torch.cat(representation_heads, dim=-1)
        embeddings = self.embed_output(representations)
        embeddings = self.output_dropout(embeddings)

        return embeddings


class SwinSelfAttention(nn.Module):
    """Implements locality self attention from https://arxiv.org/pdf/2112.13492.pdf"""

    _bias_index_pairs = torch.tensor(
        [[i, j] for i in range(parameters.WINDOW_PATCH_SHAPE[0]) for j in range(parameters.WINDOW_PATCH_SHAPE[1])],
        requires_grad=False,
    )
    _mask_view_indices = _bias_index_pairs[:, None, :] - _bias_index_pairs[None, :, :] + parameters.WINDOW_PATCH_SHAPE[0] - 1
    _position_bias_indices = _mask_view_indices[:, :, 0], _mask_view_indices[:, :, 1]

    @property
    def position_bias_indices(self):
        """Static class method to get the position bias indices.
        We use the same window shape for all layers, so no need to re-calculate.
        The position bias itself is a tensor of size (2 * patch_shape[1] - 1, 2 * patch_shape[0] - 1)
        We want to view it as a relative matrix in the full attention dimension
        """
        return type(self)._position_bias_indices

    def generate_mask(self, window_shape: int, patch_shape: int, apply_shift: tuple = None):
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
        )

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
        apply_shift: tuple,
        config: parameters.BrazenParameters
    ):
        super().__init__()

        position_bias_dim = (2 * config.window_patch_shape[1] - 1, 2 * config.window_patch_shape[0] - 1)
        mask = self.generate_mask(window_shape, config.window_patch_shape, apply_shift=apply_shift)
        self.attention = MultiHeadAttention(
            embedding_dim,
            config,
            mask=mask,
            position_bias_dim=position_bias_dim,
            position_bias_indices=self.position_bias_indices,
        )

    def forward(self, patches: torch.Tensor):
        query_key_value = {QUERY: patches, KEY: patches, VALUE: patches}
        output = self.attention(query_key_value)

        return output


class SwinTransformerBlock(nn.Module):
    """Applies a single layer of the transformer to the input"""

    def __init__(
        self,
        embedding_dim: int,
        patch_dim: int,
        apply_shift: bool,
        config: parameters.BrazenParameters
    ):
        super().__init__()

        for index, _ in enumerate(config.image_shape):
            assert (
                config.window_patch_shape[0] % 2 == 0 and config.window_patch_shape[1] % 2 == 0
            ), "Window shape must be even for even window splitting"
            assert config.image_shape[index] % patch_dim == 0, "Image width must be divisible by patch dimension"
            assert (config.image_shape[index] // patch_dim) % config.window_patch_shape[
                index
            ] == 0, "Number of patches must be divisible by window dimension"

        self.cyclic_shift = (
            (
                -(config.window_patch_shape[1] // 2),
                -(config.window_patch_shape[0] // 2),
            ) # use (vertical, horizontal) ordering to match embedding
            if apply_shift is True
            else None
        )

        window_shape = (
            config.image_shape[0] // patch_dim // config.window_patch_shape[0],
            config.image_shape[1] // patch_dim // config.window_patch_shape[1],
        ) 
        
        self.partition_windows = einops_torch.Rearrange(
            utils.assemble_einops_string(
                input_shape="batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch",
                output_shape="batch vertical_windows horizontal_windows (vertical_patches horizontal_patches) patch",
            ),
            vertical_patches=config.window_patch_shape[1],
            horizontal_patches=config.window_patch_shape[0],
        )  # fix the number of patches per window, let it find number of windows from image

        attention = OrderedDict()
        self.self_attention_norm = nn.LayerNorm(embedding_dim)
        self.self_attention = SwinSelfAttention(embedding_dim, window_shape, apply_shift, config)

        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        feed_forward = OrderedDict()
        feed_forward["linear_1"] = nn.Linear(embedding_dim, embedding_dim * config.feed_forward_expansion)
        feed_forward["gelu"] = nn.GELU()
        feed_forward["linear_2"] = nn.Linear(embedding_dim * config.feed_forward_expansion, embedding_dim)
        feed_forward["dropout"] = nn.Dropout(config.dropout_rate)
        self.feed_forward = nn.Sequential(feed_forward)

        self.join_windows = einops_torch.Rearrange(
            utils.assemble_einops_string(
                input_shape="batch vertical_windows horizontal_windows (vertical_patches horizontal_patches) patch",
                output_shape="batch (vertical_patches vertical_windows) (horizontal_patches horizontal_windows) patch",
            ),
            vertical_windows=window_shape[1],
            vertical_patches=config.window_patch_shape[1],
        )

    def forward(self, embeddings: torch.Tensor):
        if self.cyclic_shift:
            embeddings = embeddings.roll(self.cyclic_shift, dims=(1, 2))  # base[:][0][0]  == shifted[:][4][4]

        patches = self.partition_windows(embeddings)
        normalized_patches = self.self_attention_norm(patches)
        self_attention = normalized_patches + self.self_attention(normalized_patches)

        normalized_feed_forward = self.feed_forward_norm(self_attention)
        output = normalized_feed_forward + self.feed_forward(normalized_feed_forward)

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
        embedding_dim:int,
        input_dim:int,
        num_blocks:int,
        apply_merge:bool,
        patch_dim:int,
        config: parameters.BrazenParameters
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
                vertical_reduce=config.reduce_factor,
                horizontal_reduce=config.reduce_factor,
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
                config=config
            )

        self.transform = nn.Sequential(transform_pipeline)

    def forward(self, patches: torch.Tensor):
        embeddings = self.preprocess(patches)
        transformed = self.transform(embeddings)

        return transformed


class DecoderBlock(nn.Module):
    """ Decode all of the tokens in the output sequence as well as the Swin self-attention output
    """

    def __init__(self, config: parameters.BrazenParameters):
        super().__init__()

        sequence_mask = torch.tensor(np.triu(np.full((config.sequence_length, config.sequence_length), True), 1).astype(np.bool))

        self.sequence_norm = nn.LayerNorm(config.decoder_embedding_dim)
        self.attend_sequence = MultiHeadAttention(config.decoder_embedding_dim, config, mask=sequence_mask)

        self.transform_norm = nn.LayerNorm(config.decoder_embedding_dim)
        self.transform = MultiHeadAttention(config.decoder_embedding_dim, config)

        self.feed_forward_norm = nn.LayerNorm(config.decoder_embedding_dim)
        feed_forward = OrderedDict()
        feed_forward["linear_0"] = nn.Linear(config.decoder_embedding_dim, config.decoder_embedding_dim * config.decoder_feed_forward_expansion)
        feed_forward["gelu"] = nn.GELU()
        feed_forward["linear_1"] = nn.Linear(config.decoder_embedding_dim * config.decoder_feed_forward_expansion, config.decoder_embedding_dim)
        feed_forward["dropout"] = nn.Dropout(config.dropout_rate)
        self.feed_forward = nn.Sequential(feed_forward)

    def forward(self, embeddings: dict):
        assert ENCODER in embeddings and DECODER in embeddings, "Embeddings dictionary missing keys {ENCODER} and {DECODER}"

        normalized_sequence_embeddings = self.sequence_norm(embeddings[DECODER])
        sequence_embeddings = {QUERY: normalized_sequence_embeddings, KEY: normalized_sequence_embeddings, VALUE: normalized_sequence_embeddings}
        sequence_attention = normalized_sequence_embeddings + self.attend_sequence(sequence_embeddings)

        normalized_transform_embeddings = self.transform_norm(sequence_attention) # I'm not normalizing the encoder embedidngs since they're "singleton". Skip post-normalize
        transform_embeddings = {QUERY: normalized_transform_embeddings, KEY: embeddings[ENCODER], VALUE: embeddings[ENCODER]}
        transform_attention = normalized_transform_embeddings + self.transform(transform_embeddings)

        normalized_feed_forward_embeddings = self.feed_forward_norm(transform_attention)
        feed_forward = normalized_feed_forward_embeddings + self.feed_forward(normalized_feed_forward_embeddings)

        return {DECODER: feed_forward, ENCODER: embeddings[ENCODER]}


class BrazenNet(nn.Module):
    """A perception stack consisting of a shifted-window transformer stage and a feed-forward layer
    Predicts the entire output sequence in one go
    """

    def __init__(
        self,
        config: parameters.BrazenParameters,
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
                config=config
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

        self.output_length = config.sequence_length
        self.embed_tokens = nn.Embedding(
            config.total_symbols, config.decoder_embedding_dim, padding_idx=config.padding_symbol
        )
        self.embed_positions = nn.Embedding(config.sequence_length, config.decoder_embedding_dim)

        decoder = OrderedDict()
        for index in range(config.num_decoder_blocks):
            decoder["block_{}".format(index)] = DecoderBlock(config)
        self.decoder = nn.Sequential(decoder)

        # Map transformer outputs to sequence of symbols
        output = OrderedDict()
        output["linear_out"] = nn.Linear(config.decoder_embedding_dim, config.total_symbols)
        self.output = nn.Sequential(output)

    def get_parameters(self):
        """ Sort parameters into whether or not their weights should be decayed
        """

        parameter_decay = {True: set(), False: set()}

        for module_name, module in self.named_modules():
            for parameter_name, parameter in module.named_parameters():
                bucket = True if parameter_name.endswith("weight") and isinstance(module, nn.Linear) else False 
                parameter_decay[bucket].add(parameter)
        
        parameter_decay[False] -= parameter_decay[True] 
        model_parameters = [
            {"params": list(parameter_decay[True]), "weight_decay": self.config.weight_decay},
            {"params": list(parameter_decay[False]), "weight_decay": 0.0},
        ]

        return model_parameters
        
    def forward(self, images: torch.Tensor, labels: torch.Tensor = None):
        encoder_patches = self.extract_encoder_patches(images)
        encoded_images = self.encoder(encoder_patches)

        assert encoded_images.shape[1:3] == (
            self.config.window_patch_shape[1],
            self.config.window_patch_shape[0],
        ), "The output was not reduced to a single window at the final output stage. Check window shape, reduce factor, and encoder block stages."

        encoder_embeddings = self.embed_encoder_output(encoded_images)

        embeddings = {
            ENCODER: einops.rearrange(encoder_embeddings, "batch encoder_embedding -> batch 1 encoder_embedding").expand(self.config.batch_size, self.config.sequence_length, self.config.decoder_embedding_dim)
        }
        positions = torch.arange(self.config.sequence_length, device=encoder_embeddings.device, dtype=torch.long)

        if labels is None:  # the model is being used in inference mdoe
            labels = torch.tensor(
                [self.config.beginning_of_sequence if index == 0 else self.config.padding_symbol for index in range(self.output_length)], device=images.device
            )  # Begin masked
            batch_labels = einops.repeat(labels, "symbol -> batch symbol", batch=self.config.batch_size)
            embeddings[DECODER] = self.embed_tokens(batch_labels) + self.embed_positions(positions)

            for index in range(self.output_length):
                decoder_outputs = self.decoder(embeddings)
                output_sequence = self.output(decoder_outputs[DECODER])
                labels[: index + 1] = torch.max(output_sequence, dim=-1)[1][0, : index + 1]

            loss = None
        else:
            # Shift right
            shifted_labels = torch.roll(labels, shifts=1, dims=-1)
            shifted_labels[:, 0] = self.config.beginning_of_sequence

            embeddings[DECODER] = self.embed_tokens(shifted_labels) + self.embed_positions(positions)

            decoder_outputs = self.decoder(embeddings)
            output_sequence = self.output(decoder_outputs[DECODER])
            loss = functional.cross_entropy(output_sequence.transpose(2, 1), labels, reduction="mean")

        return output_sequence, loss
