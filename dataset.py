import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, root, phase='train', transform=None):
        self.transform = transform

        self.imagesA = sorted(glob.glob(os.path.join(root, phase + 'A', '*.jpg')))
        self.imagesB = sorted(glob.glob(os.path.join(root, phase + 'B', '*.jpg')))

    def __getitem__(self, index):
        imageA = Image.open(self.imagesA[index % len(self.imagesA)])
        imageB = Image.open(self.imagesB[index % len(self.imagesB)])

        if self.transform is not None:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)
        else:
            imageA = np.array(imageA)
            imageB = np.array(imageB)

        return imageA, imageB

    def __len__(self):
        return max(len(self.imagesA), len(self.imagesB))
