import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import CatsDogs
from model import DogsCatsModel_FCN
from torchmetrics.classification import BinaryAccuracy
from sklearn.model_selection import train_test_split
import torch.multiprocessing
from dotenv import load_dotenv
from torch import mps
import os
import gc


def train_step():
    epoch_loss = 0
    epoch_acc = 0

    for step, (x_sample, label) in enumerate(train_loader):
        x_sample = x_sample.to(device=device)
        label = label.to(device=device)

        # train the model
        logits, probabs = model(x_sample)

        model.zero_grad()
        loss = objective(logits, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Metrics
        epoch_acc += accuracy(probabs, label).item()

        del x_sample
        del label
        del loss
        del logits
        del probabs

        mps.empty_cache()
        gc.collect(generation=2)

    return epoch_loss/train_steps, epoch_acc/train_steps


def test_step():
    epoch_loss = 0
    epoch_acc = 0

    for step, (x_sample, label) in enumerate(test_loader):
        x_sample = x_sample.to(device=device)
        label = label.to(device=device)

        logits, probabs = model(x_sample)

        loss = objective(logits, label)
        epoch_loss += loss.item()

        epoch_acc += accuracy(probabs, label).item()

        del x_sample
        del label
        del loss
        del logits
        del probabs

        mps.empty_cache()
        gc.collect(generation=2)

    return epoch_loss/test_steps, epoch_acc/test_steps


def training_loop():

    for epoch in range(NUM_EPOCHS):
        model.train(True)

        train_loss, train_acc = train_step()
        mps.empty_cache()
        model.eval()

        with torch.no_grad():
            test_loss, test_acc = test_step()

            mps.empty_cache()
            print("Epoch: ", epoch+1)
            print("Train Loss: ", train_loss)
            print("Train Accuracy: ", train_acc)
            print("Test Loss: ", test_loss)
            print("Test Accuracy: ", test_acc)

            # wandb.log({
            #     "Train Loss": train_loss,
            #     "Test Loss": test_loss,
            #     "Train Accuracy": train_acc,
            #     "Test Accuracy": test_acc
            # })


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    load_dotenv("Lab_4/custom_cnn/.env")

    global_path = os.getenv("train")
    img_paths = os.listdir(global_path)

    img_paths_clean = list()

    # remove fragments
    for i in img_paths:
        if '_' not in i:
            img_paths_clean.append(i)

    # train test split
    train, test = train_test_split(
        img_paths_clean, test_size=0.25)

    # Create dataloaders
    train_set = CatsDogs(train)
    test_set = CatsDogs(test)

    params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0
    }

    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)

    # Model and parameters
    device = torch.device("cpu")
    model = DogsCatsModel_FCN().to(device=device)

    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters())
    objective = torch.nn.BCEWithLogitsLoss()

    # Metric
    accuracy = BinaryAccuracy().to(device=device)

    NUM_EPOCHS = 50

    train_steps = (len(train)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test)+params['batch_size']-1)//params['batch_size']

    training_loop()
