from numpy import savetxt, loadtxt
from torch import save
import os


class DataUtils:
    """
    Common class for data utilities.
    """

    @staticmethod
    def save_data(data_path, data_set):
        """
        Save the data and labels to a specified directory.
        :param data_path: The data path where to save the labels.
        :param data_set: The dataset to save.
        """
        save(data_set.data.unsqueeze(1), os.path.join(data_path, "_test.pth"))
        save(data_set.targets, os.path.join(data_path, "_test_labels.pth"))

    @staticmethod
    def save_indices(train_indices, test_indices, data_path):
        """
        Saves the train and test indices.
        :param train_indices: The train indices to save.
        :param test_indices: The test indices to save.
        :param data_path: The data path where the indices should be stored.
        """
        savetxt(os.path.join(data_path, 'train_indices.csv'), train_indices, delimiter=',', fmt='%i')
        savetxt(os.path.join(data_path, 'test_indices.csv'), test_indices, delimiter=',', fmt='%i')

    @staticmethod
    def load_indices(data_path):
        """
        Loads train and test indices from the csv files.
        :param data_path: The data path where the indices are stored.
        :return: The loaded train and test indices.
        """
        train_indices = loadtxt(os.path.join(data_path, 'train_indices.csv'), delimiter=',')
        test_indices = loadtxt(os.path.join(data_path, 'test_indices.csv'), delimiter=',')
        return train_indices.astype(int), test_indices.astype(int)
