from common.utils.pytorch_utils import torch2json
from mnist.common.mnist_data_loader import MnistDataLoader


class JSONGrapheneRESTMnistDataLoader(MnistDataLoader):
    """
    A Private MNIST Data loader responsible for encrypting and sharing the data.
    """

    def __init__(self, data_path, test_batch_size):
        """
        Creates a PrivateMnistDataLoader.
        """
        super(JSONGrapheneRESTMnistDataLoader, self).__init__(data_path, test_batch_size)
        self.private_test_loader = []

    def encode_test_data(self):
        """
        Encodes the test data as JSON.
        """
        for data, labels in self.test_loader:
            self.private_test_loader.append((
                torch2json(data),
                labels
            ))
