from pathlib import Path
import math
import os
import time

from matplotlib import pyplot as plt
from torch.utils import data as torchdata
from torch import cuda, optim
import torch
import numpy as np
import wandb

import dataset
import neural_network
import symposium
import parameters


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # verbose debugging
PRIMUS_PATH = Path(Path.home(), Path("Data/sheet-music/primus"))
MODEL_PATH = "./brazen-net.pth"
MODEL_FOLDER = Path("models")


def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def write_disk(image_batch, labels, name_base="brazen", output_folder="output"):
    """ """
    for index, image in enumerate(image_batch):
        plt.imshow(image)
        plt.savefig(f"{output_folder}/{name_base}_{index}.png")
        label = labels
        with open(f"{output_folder}/{name_base}_{index}.txt", "w") as file:
            file.write(" ".join(label))


def infer(model, inputs, token_map, config:parameters.BrazenParameters, labels=None):
    """ """
    outputs, loss = model(
        inputs, labels=labels
    )  # TODO: Batch size work for NLLLoss in k dimensions https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
    _, label_indices = torch.max(outputs, 2)

    labels = []
    for batch in label_indices:
        batch_labels = []
        for index in batch:
            if index == config.end_of_sequence:
                batch_labels.append("EOS")
            elif index == config.beginning_of_sequence:
                batch_labels.append("BOS")
            elif index == config.padding_symbol:
                batch_labels.append("")
            else:
                batch_labels.append(token_map[index])
        labels.append(batch_labels)

    return {"raw": outputs, "indices": label_indices, "labels": labels, "loss": loss}


def train(model, train_loader, device, token_map, config:parameters.BrazenParameters, use_wandb:bool=True):
    """Bingus"""
    optimizer = optim.AdamW(model.get_parameters(), betas=config.betas, eps=config.eps)
    model.train()

    if use_wandb:
        model_config = vars(model.config)
        model_config.pop("self")
        wandb.init(project="brazen-score", entity="msnidal", config=model_config)
        wandb.watch(model)

    for batch_index, (inputs, labels) in enumerate(train_loader):  # get index and batch
        samples_processed = batch_index * config.batch_size
        if samples_processed > config.exit_after:
            print(f"Done training after {samples_processed} samples!")
            break
        
        if batch_index % 1000 == 0:
            #print("Saving intermediate model")
            model_path = MODEL_FOLDER / "train.pth"
            torch.save(model.state_dict(), model_path)

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = infer(model, inputs, token_map, config, labels=labels)
        loss = outputs["loss"]

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
        optimizer.step()
        model.zero_grad()

        if samples_processed < config.warmup_samples:
            learning_rate = config.learning_rate * (samples_processed / config.warmup_samples)
        else:
            progress = (samples_processed - config.warmup_samples) / (config.exit_after)
            learning_rate = config.learning_rate * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        if use_wandb:
            wandb.log({"loss": loss, "batch_index": batch_index, "samples_processed": samples_processed, "learning_rate": learning_rate})



def test(model, test_loader, device, token_map, config:parameters.BrazenParameters, exit_after:int=10):
    """Test"""
    model.eval()  # eval mode

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for index, (images, labels) in enumerate(test_loader):
            # calculate outputs by running images through the network
            if index > exit_after:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = infer(model, images, token_map, config)
            predicted = outputs["indices"]

            # Comment out
            write_disk(images.cpu(), outputs["labels"])


if __name__ == "__main__":
    # Create, split dataset into train & test
    torch.manual_seed(0)
    config = parameters.BrazenParameters()

    dataset_prompt = ""
    while dataset_prompt not in ["0", "1"]:
        dataset_prompt = input("Choose between [0: primus, 1: symposium]: ")
    
    if dataset_prompt == "0":
        primus_dataset = dataset.PrimusDataset(config, PRIMUS_PATH)
        #config.set_dataset_properties(len(primus_dataset.tokens), primus_dataset.max_label_length)
        token_map = primus_dataset.tokens

        train_size = int(0.8 * len(primus_dataset))
        test_size = len(primus_dataset) - train_size
        train_dataset, test_dataset = torchdata.random_split(primus_dataset, [train_size, test_size])

        train_loader = torchdata.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
        test_loader = torchdata.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    else:
        symposium = symposium.Symposium(config)
        #config.set_dataset_properties(len(symposium.token_map), symposium.max_label_length)
        token_map = symposium.token_map

        train_loader = torchdata.DataLoader(symposium, batch_size=config.batch_size, num_workers=config.num_workers)
        test_loader = torchdata.DataLoader(symposium, batch_size=config.batch_size, num_workers=config.num_workers)


    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Using {device} device")

    trained_models = list(MODEL_FOLDER.glob("**/*.pth"))
    did_load = False
    if trained_models:
        print("Found the following models: ")
        for index, model_path in enumerate(trained_models):
            print(f"{index}\t: {model_path}")
        prompt = ""
        while prompt != "T" and prompt != "L":
            prompt = input("Enter T to train or L to load: ")
        if prompt == "L":
            while prompt not in range(len(trained_models)):
                prompt = int(input("Select the model index from above: "))
            selection = trained_models[prompt]
            print(f"Loading model {selection}...")
            model_state = torch.load(selection)
            model = neural_network.BrazenNet(config).to(device)
            model.load_state_dict(model_state)

            print("Done loading!")
            did_load = True

    if not did_load:
        print("Creating BrazenNet...")
        model = neural_network.BrazenNet(config).to(device)
        model.apply(neural_network.init_weights)
        print("Done creating!")

        print("Training model...")
        use_wandb=None
        while use_wandb not in [True, False]:
            wandb_prompt = input("Use wandb? (T/F): ")
            use_wandb = wandb_prompt == "T" if wandb_prompt in ["T", "F"] else None

        train(model, train_loader, device, token_map, config, use_wandb=use_wandb)
        print("Done training!")

        print("Saving model...")
        model_path = MODEL_FOLDER / f"{time.ctime()}.pth"
        torch.save(model.state_dict(), model_path)
        print("Done saving model!")

    test(model, test_loader, device, token_map, config)
