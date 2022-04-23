import os
import errno
import pickle5 as pickle
#import pickle
import pathlib

import pandas as pd
import torch
from torch.utils import data as torchdata
from torchvision import io as tvio
from torchvision.transforms import functional as tvfunctional
from torchvision import transforms as tvtransforms
from PIL import Image

LABEL_MODE = "semantic"
NUM_SYMBOLS = 758 if LABEL_MODE == "agnostic" else 1781
SEQUENCE_DIM = 75 if LABEL_MODE == "agnostic" else 58
RAW_IMAGE_SHAPE = (2048, 2048)
IMAGE_SHAPE = (1024, 1024)  # rough ratio that's easily dividible

TOKEN_PATH = pathlib.Path(f"tokens_{LABEL_MODE}.pickle")


class PadToLargest:
    def __init__(self, max_image_size=(0, 0), fill=0):
        self.max_image_size = max_image_size
        self.fill = fill
        self.padding_mode = "constant"

    def __call__(self, tensor):
        """Pad the given tensor on bottom and right sides to the max size with the given "pad" value"""
        delta_width = self.max_image_size[0] - tensor.shape[2]
        delta_height = self.max_image_size[1] - tensor.shape[1]
        return tvfunctional.pad(tensor, (0, 0, delta_width, delta_height), self.fill, self.padding_mode)


class PrimusDataset(torchdata.Dataset):
    def __init__(self, root_path: pathlib.Path, transforms=[]):
        if not root_path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), root_path)

        self.root_path = root_path
        self.scores = [score.name for score in root_path.iterdir() if score.is_dir()]
        self.max_label_length = self.get_max_label_length()

        self.tokens = self.get_token_mapping()

        transforms = [PadToLargest(RAW_IMAGE_SHAPE, 255), tvtransforms.Resize(IMAGE_SHAPE)] + transforms
        self.transforms = tvtransforms.Compose(transforms)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        image = self.get_score_image(self.scores[index])
        label = self.get_score_label(self.scores[index], encode_tokens=True, pad_length=True)

        return image, label

    def get_score_image(self, score):
        """Get the image for a score."""
        image_file = self.root_path / score / (score + ".png")
        image = tvio.read_image(str(image_file), tvio.ImageReadMode.GRAY)
        image = self.transforms(image)
        image = image.type(torch.FloatTensor).rename("channels", "height", "width") / 255.0  # normalize to float
        image = torch.squeeze(image, "channels")  # remove channel dimension

        return image.rename(None)

    def get_score_label(self, score, encode_tokens=False, pad_length=False):
        """Get the vectorized, agnostic label for a score."""
        agnostic_label_file = self.root_path / score / (score + ".semantic")

        with agnostic_label_file.open() as text:
            raw_label = text.read()
            label = [token for token in raw_label.split("\t") if token != ""]

        if encode_tokens:
            label = [self.tokens.index(token) for token in label]

            if pad_length:
                # special value of NUM_SYMBOLS is the empty padding value
                length_pad = [NUM_SYMBOLS for _ in range(self.max_label_length - len(label))]
                label += length_pad

            label = torch.tensor(label, dtype=torch.long)
            

        return label

    def get_max_image_size(self):
        """Get the maximum size of the score images."""
        max_image_size = [0, 0]
        for score in self.scores:
            score_path = self.root_path / score / (score + ".png")
            im = Image.open(str(score_path))
            if im.size[0] > max_image_size[0]:
                max_image_size[0] = im.size[0]
            if im.size[1] > max_image_size[1]:
                max_image_size[1] = im.size[1]

        return max_image_size

    def get_max_label_length(self):
        """Get the maximum length of the label sequence."""
        max_length = 0
        for score in self.scores:
            label = self.get_score_label(score, encode_tokens=False)
            if len(label) > max_length:
                max_length = len(label)

        return max_length

    def get_token_mapping(self):
        """Load or create the token mapping if it does not already exist."""

        if TOKEN_PATH.exists():
            with open(str(TOKEN_PATH), "rb") as handle:
                tokens = pickle.load(handle)
        else:
            tokens = []
            for score in self.scores:
                label = self.get_score_label(score, encode_tokens=False)  # we can't do this until after this step
                for token in label:
                    if token not in tokens:
                        tokens.append(token)

            with open(str(TOKEN_PATH), "wb") as handle:
                pickle.dump(tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return tokens
