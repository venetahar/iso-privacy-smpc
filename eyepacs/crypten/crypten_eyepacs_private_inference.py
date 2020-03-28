import torch
from torch import save, flatten
from torchvision.models import resnext50_32x4d

from eyepacs.common.eyepacs_data_loader import EyepacsDataLoader
from eyepacs.crypten.crypten_eyepacs import CryptenEyepacs
from eyepacs.common.constants import TEST_CSV_PATH_1

EYEPACS_MODEL_PATH = '../models/alice_eyepacs_model.pth'


def save_model(model, model_path):
    save(model, model_path)


def save_labels(data_loader, data_path):
    test_data = None
    test_labels = None
    for data, labels in data_loader:
        if test_data is None:
            test_data = data
            test_labels = labels
        else:
            test_data = torch.cat((test_data, data))
            test_labels = torch.cat((test_labels, labels))

    test_data = flatten(test_data)
    save(test_data.unsqueeze(1), data_path + "_test.pth")
    save(test_labels, data_path + "_test_labels.pth")


def evaluaute_encrypted(model_path=EYEPACS_MODEL_PATH):
    crypten_model = CryptenEyepacs()
    dummy_model = resnext50_32x4d()
    crypten_model.encrypt_evaluate_model(model_path, '../data/bob_test.pth',
                                         '../data/bob_test_labels.pth', dummy_model)


model = resnext50_32x4d(pretrained=True)
save_model(model, EYEPACS_MODEL_PATH)

eyepacs_data_loader = EyepacsDataLoader(TEST_CSV_PATH_1, TEST_CSV_PATH_1)
save_labels(eyepacs_data_loader.test_loader, '../data/bob_')

evaluaute_encrypted(EYEPACS_MODEL_PATH)
