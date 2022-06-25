from pathlib import Path
import math
import hashlib
import functools
import os
from datetime import datetime
import time

from matplotlib import pyplot as plt
from py import process
from torch.utils import data as torchdata
from torch import cuda, optim, nn, multiprocessing, distributed
import torch
import numpy as np
import wandb
import argparse

import brazen_score
import symposium
import primus
import parameters


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # verbose debugging
os.environ['MASTER_ADDR'] = "localhost"
os.environ['MASTER_PORT'] = "8888"

PRIMUS_PATH = Path(Path.home(), Path("Data/sheet-music/primus"))
MODEL_FILENAME = "brazen-net.pth"

MODEL_FOLDER = Path("models")
PRIMUS, SYMPOSIUM = "primus", "symposium"

def init_weights(module, standard_deviation):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=standard_deviation)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -1.0, 1.0)
    elif isinstance(module, nn.parameter.Parameter):
        nn.init.trunc_normal_(module, 0.0, std=standard_deviation)


def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def validate(date_text, format_template):
    try:
        datetime.strptime(date_text, format_template)
    except ValueError:
        return False
    return True


def write_disk(image_batch, labels, name_base="brazen", output_folder="output"):
    """ """
    for index, image in enumerate(image_batch):
        plt.imshow(image)
        plt.savefig(f"{output_folder}/{name_base}_{index}.png")
        label = labels[index]
        with open(f"{output_folder}/{name_base}_{index}.txt", "w") as file:
            file.write(" ".join(label))

def generate_label(label_indices, token_map, config, labels=None):
    output_labels = []
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
        output_labels.append(batch_labels)

    accuracy = (labels == label_indices).sum().item() / (label_indices.size()[0] * label_indices.size()[1]) if labels is not None else None
    return output_labels, accuracy

def infer(model, inputs, token_map, config:parameters.BrazenParameters, labels=None):
    """ """
    outputs, loss = model(
        inputs, labels=labels
    )  # TODO: Batch size work for NLLLoss in k dimensions https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
    _, label_indices = torch.max(outputs, dim=-1)

    output_labels, accuracy = generate_label(label_indices, token_map, config, labels)

    return {"raw": outputs, "indices": label_indices, "labels": output_labels, "loss": loss, "accuracy": accuracy}


def train(model, train_loader, device, token_map, config:parameters.BrazenParameters, use_wandb:bool=True):
    """Bingus"""
    optimizer = optim.AdamW(model.get_parameters(), betas=config.betas, eps=config.eps)
    model.train()

    if use_wandb:
        model_config = vars(model.config)
        model_config.pop("self")
        wandb.init(project="brazen-score", entity="msnidal", config=model_config)
        wandb.watch(model)

    running_accuracy, running_loss = 0, 0
    for batch_index, (inputs, labels) in enumerate(train_loader):  # get index and batch
        samples_processed = batch_index * config.batch_size
        if samples_processed > config.exit_after:
            print(f"Done training after {samples_processed} samples!")
            break
        
        if (batch_index + 1) % config.save_every == 0:
            model_path = MODEL_FOLDER / "train.pth"
            torch.save(model.state_dict(), model_path)

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = infer(model, inputs, token_map, config, labels=labels)
        running_accuracy += outputs["accuracy"]
        loss = outputs["loss"]
        loss.backward()

        running_loss += loss

        if (batch_index + 1) % config.optimize_every == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()
            optimizer.zero_grad()

            wind_down_samples = config.exit_after - config.warmup_samples
            if samples_processed < config.warmup_samples:
                learning_rate = (samples_processed / config.warmup_samples) * config.learning_rate
            else:
                progress = float(samples_processed - config.warmup_samples) / float(max(1, wind_down_samples))
                learning_rate = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress))) * config.learning_rate

            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            accuracy = running_accuracy / config.optimize_every
            running_accuracy = 0

            loss = running_loss / config.optimize_every
            running_loss = 0

            if use_wandb:
                wandb.log({"loss": loss, "batch_index": batch_index, "samples_processed": samples_processed, "learning_rate": learning_rate, "accuracy": accuracy})
            



def test(model, test_loader, device, token_map, config:parameters.BrazenParameters, exit_after:int=0):
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

def main(process_index, args):
    # Create, split dataset into train & test
    distributed.init_process_group(
        backend='nccl',
        init_method="env://",
        world_size=args.gpus,
        rank=process_index
    )

    torch.manual_seed(args.seed)
    seed = process_index + args.seed
    config = parameters.BrazenParameters(seed=seed, model_loaded=args.load_file, dataset=args.dataset)

    if args.dataset == PRIMUS:
        primus_dataset = primus.PrimusDataset(config, PRIMUS_PATH)
        token_map = primus_dataset.tokens

        train_size = int(0.8 * len(primus_dataset))
        test_size = len(primus_dataset) - train_size
        train_dataset, test_dataset = torchdata.random_split(primus_dataset, [train_size, test_size])

        train_loader = torchdata.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        test_loader = torchdata.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    else:
        symposium_dataset = symposium.Symposium(config)
        token_map = symposium_dataset.token_map

        train_loader = torchdata.DataLoader(symposium_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
        test_loader = torchdata.DataLoader(symposium_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if args.load_file is not None:
        model_path = MODEL_FOLDER / str(args.load_file)
        print(f"Loading model {model_path}...")
        model_state = torch.load(model_path, map_location=device)
        model = brazen_score.BrazenScore(config).to(device)
        model.load_state_dict(model_state)

        print("Done loading!")
    else:
        print("Creating BrazenScore...")
        model = brazen_score.BrazenScore(config).to(device)
        configured_init_weights = functools.partial(init_weights, standard_deviation=config.standard_deviation)
        model.apply(configured_init_weights)
        print("Done creating!")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[process_index])
    
    if args.mode == "train":
        print("Training model...")
        train(model, train_loader, device, token_map, config, use_wandb=args.track)
        print("Done training!")

        print("Saving model...")
        model_path = MODEL_FOLDER / args.save_file
        torch.save(model.state_dict(), model_path)
        print("Done saving model!")
    else:
        print("Inferring...")
        test(model, test_loader, device, token_map, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains the brazen-score model")
    parser.add_argument("mode", choices=["train", "test"], help="Train from scratch or test an existing model")
    parser.add_argument("--load-file", help=f"Filename to the load the model for testing or training, within the folder {MODEL_FOLDER}")
    parser.add_argument("--save-file", default=MODEL_FILENAME, help=f"Filename to save the model upon completion of training, within the folder {MODEL_FOLDER}")
    parser.add_argument("--dataset", default=SYMPOSIUM, choices=[SYMPOSIUM, PRIMUS], help="Which of the two datasets to for training or evaluation")
    parser.add_argument("--seed", default=0, help="Torch manual seed to set before training")
    parser.add_argument("--track", action="store_const", const=True, default=False, help="Track the experiment using weights & biases")
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs to use for training. Spawns a process using torch.multiprocessing and DistributedDataParallel")
    args = parser.parse_args()

    multiprocessing.spawn(main, nprocs=args.gpus, args=(args,))
