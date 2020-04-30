from common.model_training.model_training import ModelTraining
from malaria.common.constants import TRAINING_PARAMS, TEST_BATCH_SIZE
from malaria.common.conv_pool_model import ConvPoolModel
from malaria.common.malaria_data_loader import MalariaDataLoader
import torch


def train_malaria_model(model_path, data_path):
    """
    Trains a malaria model.
    :param model_path: The model path.
    :param data_path: The data path.
    """
    data_loader = MalariaDataLoader(data_path, TEST_BATCH_SIZE, should_load_split=False)
    model = ConvPoolModel(input_shape=(32, 32, 3), num_classes=2, conv_kernel_sizes=[5, 5], channels=[36, 36],
                          avg_pool_sizes=[2, 2], fc_units=[72])
    malaria_training = ModelTraining(model, data_loader, training_parameters=TRAINING_PARAMS)
    malaria_training.train()
    malaria_training.evaluate_plain_text()
    malaria_training.save_model(model_path)


def measure_malaria_plain_text_runtime(model_path, data_path, num_runs=20):
    mnist_data_loader = MalariaDataLoader(data_path, 1, False)
    model = torch.load(model_path)
    model_training = ModelTraining(model, mnist_data_loader, training_parameters=TRAINING_PARAMS)
    model_training.measure_plaintext_runtime(num_runs)


def evaluate_saved_model(model_path, data_loader):
    """
    Evaluates a saved models in plain text.
    :param model_path: The model path.
    :param data_loader: The data loader.
    """
    model = torch.load(model_path)
    malaria_training = ModelTraining(model, data_loader, training_parameters=TRAINING_PARAMS)
    malaria_training.evaluate_plain_text()

