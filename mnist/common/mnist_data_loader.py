from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets

from mnist.common.constants import BATCH_SIZE, TEST_BATCH_SIZE


class MnistDataLoader:
    """
    A simple MNIST data loader.
    """

    def __init__(self, data_path):
        """
        Creates an MnistDataLoader which has a train and a test loader.
        """
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=self.transform)
        self.train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        self.test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(self.test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)
