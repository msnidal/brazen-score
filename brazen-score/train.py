from pathlib import Path
import collections

from dataset import PrimusDataset
from neural_network import BrazenNet

BATCH_SIZE = 16 # deal with this later
PRIMUS_PATH = Path(Path.home(), Path("Data/sheet-music/primus"))
MODEL_PATH = "./brazen-net.pth"

from matplotlib import pyplot as plt
from torch.utils import data as torchdata
from torch import cuda, nn, optim
import torch


if __name__ == "__main__":
  # Create, split dataset into train & test
  primus_dataset = PrimusDataset(PRIMUS_PATH)
  train_size = int(0.8 * len(primus_dataset))
  test_size = len(primus_dataset) - train_size
  train_dataset, test_dataset = torchdata.random_split(primus_dataset, [train_size, test_size])
  train_loader = torchdata.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = torchdata.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

  device = "cuda" if cuda.is_available() else "cpu"
  print(f"Using {device} device")

  net = BrazenNet().to(device)
  loss_function = nn.NLLLoss()
  optimizer = optim.Adam(net.parameters(), lr=1e-3)

  train_length = len(train_dataset)
  print("Train length:", train_length)
  net.train() # set training mode

  for index, (X, y) in enumerate(train_loader): #get index and batch
    optimizer.zero_grad()
    pred = net(X)
    #_, prediction_indexes = torch.max(pred, 1) # tuple is values, indices. we want indices
    prediction = pred.transpose(1, 2)

    loss = loss_function(prediction, y)
    loss.backward()
    optimizer.step()

    if index % 100 == 0:
      loss, current = loss.item(), BATCH_SIZE * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{train_length:>5d}]")
  
  torch.save(net.state_dict(), MODEL_PATH)