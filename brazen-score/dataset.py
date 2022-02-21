import os
import pandas as pd
import errno 

from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path

#SCORES_ROOT = "~/Data/sheet-music/primus/"

class PrimusDataset(Dataset):
    def __init__(self, root_path: Path, transform=None, target_transform=None):
        if not root_path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), root_path)
        self.root_path = root_path
        self.scores = [score.name for score in root_path.iterdir() if score.is_dir()]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        score_path = self.root_path / self.scores[index]
        image_file = score_path / (self.scores[index] + ".png")
        image = read_image(str(image_file))
        text_file = score_path / (self.scores[index] + ".agnostic")

        with text_file.open() as text:
            label = text.read()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label