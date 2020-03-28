from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from eyepacs.common.constants import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LABELS
from eyepacs.common.image_label_dataset import ImageLabelDataset


class EyepacsDataLoader:

    def __init__(self, train_csv_path, test_csv_path):
        norm_mean = [0.42, 0.22, 0.075]  # [0.485, 0.456, 0.406]
        norm_std = [0.27, 0.15, 0.081]  # [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize(540),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

        train_set = ImageLabelDataset(csv=train_csv_path, shuffle=False, transform=self.transform, label_names=LABELS)
        self.train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)

        self.test_set = ImageLabelDataset(csv=test_csv_path, shuffle=False, transform=self.transform, label_names=LABELS)
        self.test_loader = DataLoader(self.test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)
