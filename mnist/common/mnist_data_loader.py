from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from mnist.common.constants import BATCH_SIZE


def mnist_transform():
    return Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])


class MnistDataLoader:
    """
    A simple MNIST data loader.
    """

    def __init__(self, data_path, test_batch_size):
        """
        Creates an MnistDataLoader which has a train and a test loader.
        """
        self.transform = mnist_transform()

        self.test_batch_size = test_batch_size

        train_set = MNIST(root=data_path, train=True, download=True, transform=self.transform)
        self.train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        self.test_set = MNIST(root=data_path, train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(self.test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)
