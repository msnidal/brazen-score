from pathlib import Path
import collections
import os

from dataset import PrimusDataset, NUM_SYMBOLS
import neural_network

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # verbose debugging
BATCH_SIZE = 8
PRIMUS_PATH = Path(Path.home(), Path("Data/sheet-music/primus"))
MODEL_PATH = "./brazen-net.pth"
EPOCH_SIZE = 100

from matplotlib import pyplot as plt
from torch.utils import data as torchdata
from torch import cuda, nn, optim
import torch
import numpy as np

def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def write_disk(image_batch, labels, name_base="brazen", output_folder="output"):
    """ """
    grayscale_images = [image_channels[0] for image_channels in image_batch]
    for index, image in enumerate(grayscale_images):
        plt.imshow(image)
        plt.savefig(f"{output_folder}/{name_base}_{index}.png")
        label = labels[index]
        with open(f"{output_folder}/{name_base}_{index}.txt", "w") as file:
            file.write(" ".join(label))


def infer(model, inputs, token_map, labels=None):
    """ """
    outputs, loss = model(
        inputs, labels=labels
    )  # TODO: Batch size work for NLLLoss in k dimensions https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
    _, label_indices = torch.max(outputs, 2)

    labels = []
    for batch in label_indices:
        batch_labels = []
        for index in batch:
            if index == NUM_SYMBOLS:
                batch_labels.append("")
            else:
                batch_labels.append(token_map[index])
        labels.append(batch_labels)

    return {"raw": outputs[0], "indices": label_indices[0], "labels": labels[0], "loss": loss}


def train(model, train_loader, train_length, device, token_map):
    """Bingus"""
    #loss_function = nn.NLLLoss(ignore_index=NUM_SYMBOLS)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_length = len(train_dataset)
    model.train()
    training_loss = []
    epoch_loss = []

    for index, (inputs, labels) in enumerate(train_loader):  # get index and batch
        # if index == 4186:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = infer(model, inputs, token_map, labels=labels)

        prediction = outputs["raw"]
        loss = outputs["loss"]
        #loss = loss_function(prediction, labels[0])

        if index % EPOCH_SIZE == 0:
            epoch_loss_average = sum(epoch_loss) / len(epoch_loss)
            epoch_loss.clear()
            training_loss.append(epoch_loss_average)
            print(
                f"Loss: {epoch_loss_average:>7f}\t[{index * BATCH_SIZE:>5d}/{train_length:>5d}]"
            )

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())



def test(model, test_loader, device, token_map):
    """Test"""
    model.eval()  # eval mode

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for index, (images, labels) in enumerate(test_loader):
            # calculate outputs by running images through the network
            if index == 1:
                images, labels = images.to(device), labels.to(device)
                outputs = infer(model, images, token_map)
                predicted = outputs["indices"]

                # Comment out
                write_disk(images.cpu(), outputs["labels"])


if __name__ == "__main__":
    # Create, split dataset into train & test
    torch.manual_seed(0)
    primus_dataset = PrimusDataset(PRIMUS_PATH)
    token_map = primus_dataset.tokens
    train_size = int(0.8 * len(primus_dataset))
    test_size = len(primus_dataset) - train_size
    train_dataset, test_dataset = torchdata.random_split(
        primus_dataset, [train_size, test_size]
    )

    train_length = len(train_dataset)

    train_loader = torchdata.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torchdata.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Using {device} device")

    print("Creating BrazenNet...")
    model = neural_network.BrazenNet().to(device)
    print("Done creating!")
    load_model = False

    if load_model:
        print("Loading model...")
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Done loading!")
    else:
        print("Training model...")
        train(model, train_loader, train_length, device, token_map)
        print("Done training!")
        print("Saving model...")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Done saving model!")

    test(model, test_loader, device, token_map)
