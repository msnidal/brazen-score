from typing import OrderedDict

from einops.layers import torch as einops_torch
import einops
import torch
from torch import nn
from torch.nn import functional

import utils
import parameters
import models


class BrazenScore(nn.Module):
    """A perception stack consisting of a shifted-window transformer stage and a feed-forward layer
    Predicts the entire output sequence in one go
    """

    def __init__(
        self,
        config: parameters.BrazenParameters = parameters.BrazenParameters(),
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
            encoder[f"stage_{index}"] = models.SwinTransformerStage(
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
            config.total_symbols, config.decoder_embedding_dim
        )
        #self.embed_positions = nn.Embedding(config.sequence_length, config.decoder_embedding_dim)

        decoder = OrderedDict()
        for index in range(config.num_decoder_blocks):
            decoder["block_{}".format(index)] = models.DecoderBlock(config)
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
            models.ENCODER: einops.rearrange(encoder_embeddings, "batch encoder_embedding -> batch 1 encoder_embedding").expand(self.config.batch_size, self.config.sequence_length, self.config.decoder_embedding_dim)
        }
        #positions = torch.arange(self.config.sequence_length, device=encoder_embeddings.device, dtype=torch.long)

        if labels is None:  # the model is being used in inference mdoe
            labels = torch.tensor(
                [self.config.beginning_of_sequence if index == 0 else self.config.padding_symbol for index in range(self.output_length)], device=images.device
            )  # Begin masked
            labels = einops.repeat(labels, "symbol -> batch symbol", batch=self.config.batch_size)

            for index in range(self.output_length):
                embeddings[models.DECODER] = self.embed_tokens(labels) #+ self.embed_positions(positions)

                decoder_outputs = self.decoder(embeddings)
                output_sequence = self.output(decoder_outputs[models.DECODER])
                labels = torch.max(output_sequence, dim=-1)[1]
                labels = torch.roll(labels, shifts=1, dims=-1)
                labels[:, 0] = self.config.beginning_of_sequence

            loss = None
        else:
            # Shift right
            shifted_labels = torch.roll(labels, shifts=1, dims=-1)
            shifted_labels[:, 0] = self.config.beginning_of_sequence

            embeddings[models.DECODER] = self.embed_tokens(shifted_labels) #+ self.embed_positions(positions)

            decoder_outputs = self.decoder(embeddings)
            output_sequence = self.output(decoder_outputs[models.DECODER])
            loss = functional.cross_entropy(output_sequence.transpose(2, 1), labels, reduction="mean")

        return output_sequence, loss
