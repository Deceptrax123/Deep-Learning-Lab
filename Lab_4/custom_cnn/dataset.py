import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
from dotenv import load_dotenv


class CatsDogs(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        load_dotenv("Lab_4/custom_cnn/.env")
        global_path = os.getenv("train")

        img_name = self.paths[idx]

        # Load Image
        img = Image.open(global_path+img_name)

        # Transforms
        transforms = T.Compose([T.Resize(size=(128, 128)), T.ToTensor(
        ), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img_tensor = transforms(img)

        # Get Label
        if 'dog' in img_name:
            label = torch.ones(1)
        else:
            label = torch.zeros(1)

        return img_tensor, label
