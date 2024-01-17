import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np


class PCam(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        train_img = self.imgs[idx]
        train_label = self.labels[idx]

        # Transform image
        transform = T.Compose([
            T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img_tensor = transform(train_img)

        # Labels
        label_tensor = torch.from_numpy(np.reshape(
            train_label, (1,)))
        label_tensor = label_tensor.float()

        return img_tensor, label_tensor
