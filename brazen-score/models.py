import math
from typing import OrderedDict

from einops.layers import torch as einops_torch
import einops
import torch
from torch import embedding, nn
import numpy as np

import utils
import parameters

KEY = "key"
QUERY = "query"
VALUE = "value"
ENCODER = "encoder"
DECODER = "decoder"


class MultiLayerPerceptron(nn.Module):
    """ Common MLP head used for both the encoder and decoder blocks """

    def __init__(self, embedding_dim: int, config: parameters.BrazenParameters):
        super().__init__()
        feed_forward = OrderedDict()
        feed_forward["linear_1"] = nn.Linear(embedding_dim, embedding_dim * config.feed_forward_expansion)
        feed_forward["gelu"] = nn.GELU()
        feed_forward["linear_2"] = nn.Linear(embedding_dim * config.feed_forward_expansion, embedding_dim)
        feed_forward["dropout"] = nn.Dropout(config.dropout_rate)
        self.feed_forward = nn.Sequential(feed_forward)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        embeddings = self.feed_forward(input_tensor)
        return embeddings


class MultiHeadAttention(nn.Module):
    """ Generic multi-head attention module implemented throughout both visual and output sequence transformer """

    def __init__(
        self,
        embedding_dim: int,
        config: parameters.BrazenParameters,
        mask: torch.Tensor = None,
        position_bias_dim: tuple = None,
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

    def forward(self, embeddings: dict):
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
                    attention_scores + self.position_bias[head_index][self.position_bias_indices[0], self.position_bias_indices[1]]
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

        position_bias_dim = (config.num_heads, 2 * config.window_patch_shape[1] - 1, 2 * config.window_patch_shape[0] - 1)
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

        self.self_attention_norm = nn.LayerNorm(embedding_dim)
        self.self_attention = SwinSelfAttention(embedding_dim, window_shape, apply_shift, config)

        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = MultiLayerPerceptron(embedding_dim, config)

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
            embeddings = embeddings.roll(self.cyclic_shift, dims=(1, 2))

        patches = self.partition_windows(embeddings)
        normalized_patches = self.self_attention_norm(patches)
        self_attention = patches + self.self_attention(normalized_patches)

        normalized_feed_forward = self.feed_forward_norm(self_attention)
        output = self_attention + self.feed_forward(normalized_feed_forward)

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
        self.feed_forward = MultiLayerPerceptron(config.decoder_embedding_dim, config)

    def forward(self, embeddings: dict):
        assert ENCODER in embeddings and DECODER in embeddings, "Embeddings dictionary missing keys {ENCODER} and {DECODER}"

        normalized_sequence_embeddings = self.sequence_norm(embeddings[DECODER])
        sequence_embeddings = {QUERY: normalized_sequence_embeddings, KEY: normalized_sequence_embeddings, VALUE: normalized_sequence_embeddings}
        sequence_attention = embeddings[DECODER] + self.attend_sequence(sequence_embeddings)

        normalized_transform_embeddings = self.transform_norm(sequence_attention) # I'm not normalizing the encoder embedidngs since they're "singleton". Skip post-normalize
        transform_embeddings = {QUERY: normalized_transform_embeddings, KEY: embeddings[ENCODER], VALUE: embeddings[ENCODER]}
        transform_attention = sequence_attention + self.transform(transform_embeddings)

        normalized_feed_forward_embeddings = self.feed_forward_norm(transform_attention)
        feed_forward = transform_attention + self.feed_forward(normalized_feed_forward_embeddings)

        return {DECODER: feed_forward, ENCODER: embeddings[ENCODER]}