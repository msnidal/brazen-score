from pathlib import Path
import collections

from dataset import PrimusDataset
from neural_network import BrazenNet

BATCH_SIZE = 16 # deal with this later
PRIMUS_PATH = Path(Path.home(), Path("primus"))
MODEL_PATH = "./brazen-net.pth"

from matplotlib import pyplot as plt
from torch.utils import data as torchdata
from torch import cuda, nn, optim
import torch

def train(model, train_loader, train_length):
  """ Bingus """
  loss_function = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  train_length = len(train_dataset)
  print("Train length:", train_length)
  model.train()

  for index, (inputs, labels) in enumerate(train_loader): #get index and batch
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    pred = model(inputs)
    #_, prediction_indexes = torch.max(pred, 1) # tuple is values, indices. we want indices
    prediction = pred.transpose(1, 2)
    loss = loss_function(prediction, labels)

    if index % 100 == 0:
      print(f"Loss: {loss.item():>7f}\t[{index * BATCH_SIZE:>5d}/{train_length:>5d}]")
  
    loss.backward()
    optimizer.step()


def test(model, test_loader):
  """ Test """
  model.eval() # eval mode

  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for (images, labels) in test_loader:
          # calculate outputs by running images through the network
          outputs = model(images)
          # the class with the highest energy is what we choose as prediction
          junk, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  # need to change this
  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

  return model

if __name__ == "__main__":
  # Create, split dataset into train & test
  primus_dataset = PrimusDataset(PRIMUS_PATH)
  train_size = int(0.8 * len(primus_dataset))
  test_size = len(primus_dataset) - train_size
  train_dataset, test_dataset = torchdata.random_split(primus_dataset, [train_size, test_size])

  train_length = len(train_dataset)
  print("Train length:", train_length)

  train_loader = torchdata.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = torchdata.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

  device = "cuda" if cuda.is_available() else "cpu"
  print(f"Using {device} device")

  model = BrazenNet().to(device)
  train(model, train_loader, train_length)
  torch.save(model.state_dict(), MODEL_PATH)
  test(model, test_loader)