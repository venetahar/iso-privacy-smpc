from crypten import cryptensor

from mnist.common.mnist_data_loader import MnistDataLoader


class PrivateCryptenMnistDataLoader(MnistDataLoader):
    """
    A Private MNIST Data loader responsible for encrypting and sharing the data.
    """

    def __init__(self, data_path, test_batch_size):
        """
        Creates a PrivateMnistDataLoader.
        """
        super(PrivateCryptenMnistDataLoader, self).__init__(data_path, test_batch_size)
        self.private_test_loader = []

    def encrypt_test_data(self, data_owner):
        """
        Encrypts the test data.
        """
        for data, labels in self.test_loader:
            self.private_test_loader.append((
                cryptensor(data, src=data_owner),
                cryptensor(labels, src=data_owner),
            ))
