import os
import pandas as pd
import errno 

from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import pad
from pathlib import Path

class PadToLargest(object):
  def __init__(self, max_size=[0, 0]):
    self.max_size = max_size
    self.fill = 0
    self.padding_mode = "constant"
    
  def __call__(self, tensor):
    delta_width = self.max_size[0] - tensor.shape[2]
    delta_height = self.max_size[1] - tensor.shape[1]
    return pad(tensor, (0, 0, delta_width, delta_height))


class PrimusDataset(Dataset):
    def __init__(self, root_path: Path):
        if not root_path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), root_path)
        self.root_path = root_path
        self.scores = [score.name for score in root_path.iterdir() if score.is_dir()]
        self.max_size = [0, 0]

        """
        for score in self.scores:
            score_path = self.root_path / score / (score + ".png")
            im = Image.open(str(score_path))
            if im.size[0] > self.max_size[0]:
                self.max_size[0] = im.size[0]
            if im.size[1] > self.max_size[1]:
                self.max_size[1] = im.size[1]
        """
        self.max_size = [2003, 288]
            
        self.transform = PadToLargest(self.max_size)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        score_path = self.root_path / self.scores[index]
        image_file = score_path / (self.scores[index] + ".png")
        image = read_image(str(image_file), ImageReadMode.GRAY)
        text_file = score_path / (self.scores[index] + ".agnostic")

        with text_file.open() as text:
            label = text.read()

        if self.transform:
            image = self.transform(image)
        return image, label