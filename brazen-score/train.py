from dataset import PrimusDataset
from matplotlib import pyplot as plt
from pathlib import Path
from torchvision import transforms
import torch

primus_path = Path(Path.home(), Path("Data/sheet-music/primus"))

if __name__ == "__main__":
  primus_dataset = PrimusDataset(primus_path)
  score = primus_dataset[0]
  train_size = int(0.8 * len(primus_dataset))
  test_size = len(primus_dataset) - train_size
  train_dataset, test_dataset = torch.utils.data.random_split(primus_dataset, [train_size, test_size])
  print("DONE!")