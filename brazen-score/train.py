from pathlib import Path
import collections

from dataset import PrimusDataset
from neural_stack import STACK

BATCH_SIZE = 32
PRIMUS_PATH = Path(Path.home(), Path("Data/sheet-music/primus"))


from matplotlib import pyplot as plt
from torch.utils import data as torchdata


if __name__ == "__main__":
  primus_dataset = PrimusDataset(PRIMUS_PATH)
  score = primus_dataset[0]
  train_size = int(0.8 * len(primus_dataset))
  test_size = len(primus_dataset) - train_size
  train_dataset, test_dataset = torchdata.random_split(primus_dataset, [train_size, test_size])
  train_loader = torchdata.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = torchdata.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

  train_length = len(train_dataset)
  print("Train length:", train_length)
  for index, (X, y) in enumerate(train_loader): #get index and batch
    if index % 100 == 0:
      print("Epoch done")