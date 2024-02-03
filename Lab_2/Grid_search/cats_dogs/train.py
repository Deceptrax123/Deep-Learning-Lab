import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import CatsDogs
from model import DogsCatsModel
from torchmetrics.classification import BinaryAccuracy
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import torch.multiprocessing
from dotenv import load_dotenv
from torch import mps, nn
import os
import gc

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    load_dotenv("Lab_2/Grid_search/cats_dogs/.env")

    global_path = os.getenv("train")
    img_paths = os.listdir(global_path)

    img_paths_clean = list()

    # remove fragments
    for i in img_paths:
        if '_' not in i:
            img_paths_clean.append(i)

    # train test split
    train, test = train_test_split(img_paths_clean, test_size=0.25)

    # Create dataloaders

    train_set = CatsDogs(train)
    params = {
        'batch_size': 25000,  # Consider the whole dataset
        'shuffle': True,
        'num_workers': 0
    }

    train_loader = DataLoader(train_set, **params)
    # Grid Search
    model = NeuralNetClassifier(
        DogsCatsModel,
        criterion=nn.BCELoss,
        max_epochs=5,
        batch_size=8,
        verbose=False
    )
    # Define Grid Search Parameters
    param_grid = {
        'optimizer': [
            torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop
        ]
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        refit=False, scoring='accuracy', verbose=1)

    counter = 0
    search_batches = 0

    for i, data in enumerate(train_loader):
        counter += 1
        image, labels = data

        outputs = grid.fit(image, labels)
        # GridSearch for `search_batches` number of times.

        if counter == search_batches:
            break

    print('SEARCH COMPLETE')
    print("best score: {:.3f}, best params: {}".format(
        grid.best_score_, grid.best_params_))
