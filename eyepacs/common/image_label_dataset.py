# This code is copied from the following repository:
# https://github.com/amiralansary/DR-Detection/blob/master/dataset/dataset.py and belongs to the author listed below.
"""Dataset mo and prepare data for the learning pipeline"""
# Copyright (C) 2019 Amir Alansary <amiralansary@gmail.com>
# License: GPL-3.0


import pandas as pd
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


###############################################################################

class ImageLabelDataset(Dataset):
    """A dataset class to retrieve samples of paired images and labels"""

    def __init__(self, csv, shuffle=None, transform=None, label_names=None):
        """
        Args:
            csv (string): Path to the csv file with data
            shuffle (callable, optional): Shuffle list of files
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        super().__init__()
        self.transform = transform
        self.csv_file = pd.read_csv(csv)
        self.label_names = label_names
        if shuffle:
            self.csv_file = self.csv_file.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.csv_file['name'].values[idx]
        path = self.csv_file['path'].values[idx]
        label = torch.tensor(self.csv_file['level'].values[idx])
        label_hot = torch.tensor(eval(self.csv_file['level_hot'].values[idx]))
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        sample = {'name': name, 'image': image, 'label': label,
                  'label_hot': label_hot, 'path': path}

        # return sample
        return image, label

###############################################################################
def loadImageToTensor(image_file, transform=None):
    """Load an image and returns a tensor
    Args:
        image_file (string): Path to the image file
        transform (callable, optional): Optional transform to be applied  on a sample.
    """

    image = Image.open(image_file)
    if transform:
        image = transform(image)

    return image.unsqueeze(0)