import torch
from torch.utils.data import DataLoader
import torch.multiprocessing
from dataset import PCam
from torch import mps
from model import PCamModel
from torchmetrics.classification import BinaryAccuracy
import h5py
from dotenv import load_dotenv
import numpy as np
import wandb
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
        epoch_acc += accuracy(probabs.to(device='cpu'), label.to(device='cpu'))

        del x_sample
        del label
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

        epoch_acc += accuracy(probabs.to(device='cpu'), label.to(device='cpu'))

        del x_sample
        del label
        mps.empty_cache()
        gc.collect(generation=2)

    return epoch_loss/test_steps, epoch_acc/test_steps


def training_loop():

    for epoch in range(NUM_EPOCHS):
        model.train(True)

        train_loss, train_acc = train_step()
        model.eval()

        with torch.no_grad():
            test_loss, test_acc = test_step()
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

    params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0
    }
    load_dotenv("Lab_1/pcam/.env")

    train_x_h5 = os.getenv("train_path_x")
    train_y_h5 = os.getenv("train_path_y")
    test_x_h5 = os.getenv("test_path_x")
    test_y_h5 = os.getenv("test_path_y")

    # Logger
    # wandb.init(
    #     project='Pcam-classifier',
    #     config={
    #         'dataset': "PCam"
    #     }
    # )

    # convert h5 files to numpy arrays
    train_x, train_labels = np.array(
        h5py.File(train_x_h5)['x']), np.array(h5py.File(train_y_h5)['y'])
    test_x, test_labels = np.array(
        h5py.File(test_x_h5)['x']), np.array(h5py.File(test_y_h5)['y'])

    # Dataset
    train_dataset = PCam(train_x, train_labels)
    test_dataset = PCam(test_x, test_labels)

    # Data sampler
    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(test_dataset, **params)

    # Load model and parameters
    device = torch.device("mps")
    model = PCamModel().to(device=device)
    optimizer = torch.optim.Adam(params=model.parameters())

    # Losses and metrics
    objective = torch.nn.BCEWithLogitsLoss()
    accuracy = BinaryAccuracy()

    # Hyperparams
    NUM_EPOCHS = 1000

    # Steps
    train_steps = (len(train_x)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test_x)+params['batch_size']-1)//params['batch_size']

    training_loop()
