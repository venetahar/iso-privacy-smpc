from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from numpy.random import shuffle

from common.utils.data_utils import DataUtils
from malaria.common.constants import IMG_RESIZE, MALARIA_NORM_MEAN, MALARIA_NORM_STD, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, \
    TRAIN_PERCENTAGE


class MalariaDataLoader:
    """
    A data loader class for the Malaria Dataset.
    """

    def __init__(self, data_path, should_load_split):
        """
        Returns a data loader for the Malaria dataset.
        :param data_path: The path where the images are stored
        """
        self.data_path = data_path
        data_transforms = transforms.Compose([transforms.Resize(IMG_RESIZE),
                                              transforms.ToTensor(),
                                              transforms.Normalize(MALARIA_NORM_MEAN, MALARIA_NORM_STD)
                                              ])

        data = datasets.ImageFolder(self.data_path + 'cell_images', transform=data_transforms)
        train_sampler, test_sampler = self.get_samplers(data, should_load_split)

        self.train_loader = DataLoader(data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
        self.test_loader = DataLoader(data, sampler=test_sampler, batch_size=TEST_BATCH_SIZE)

    def get_samplers(self, data, should_load_split):
        """
        Returns train and test samplers.
        :param data: All available data.
        :param should_load_split: Whether it should load a preexisting data split.
        :return: The train and test samplers based on the data.
        """
        if should_load_split:
            train_indices, test_indices = DataUtils.load_indices(self.data_path)
        else:
            train_indices, test_indices = MalariaDataLoader.generate_train_test_split(data)
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        return train_sampler, test_sampler

    def generate_train_test_split(self, data):
        """
        Generates train and test splits and saves them to csv files for future use.
        :param data: The data to split.
        :return: train and test indices for the split.
        """
        num_samples = len(data)
        indices = list(range(num_samples))
        num_train_samples = int(np.floor(TRAIN_PERCENTAGE * num_samples))
        shuffle(indices)
        train_indices, test_indices = indices[:num_train_samples], indices[num_train_samples:]
        DataUtils.save_indices(train_indices, test_indices, self.data_path)
        return train_indices, test_indices
