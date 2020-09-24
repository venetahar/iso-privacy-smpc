import os

from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from malaria.common.constants import IMG_RESIZE, MALARIA_NORM_MEAN, MALARIA_NORM_STD, TRAIN_BATCH_SIZE


class MalariaDataLoader:
    """
    A data loader class for the Malaria Dataset.
    """

    def __init__(self, data_path, test_batch_size, training_folder='training', testing_folder='testing'):
        """
        Returns a data loader for the Malaria dataset.
        :param data_path: The path where the images are stored
        """
        self.data_path = data_path
        self.test_batch_size = test_batch_size
        data_transforms = transforms.Compose([transforms.Resize(IMG_RESIZE),
                                              transforms.ToTensor(),
                                              transforms.Normalize(MALARIA_NORM_MEAN, MALARIA_NORM_STD)
                                              ])

        training_cell_images_path = os.path.join(self.data_path, 'cell_images', training_folder)
        training_data = datasets.ImageFolder(training_cell_images_path, transform=data_transforms)
        self.train_loader = DataLoader(training_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)

        testing_cell_images_path = os.path.join(self.data_path, 'cell_images', testing_folder)
        testing_data = datasets.ImageFolder(testing_cell_images_path, transform=data_transforms)
        self.test_loader = DataLoader(testing_data, batch_size=test_batch_size, shuffle=True, num_workers=2)

