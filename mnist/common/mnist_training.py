import torch

from common.model_factory import ModelFactory
from common.model_training.model_training import ModelTraining
from mnist.common.constants import TRAINING_PARAMS, MNIST_DIMENSIONS, NUM_CLASSES, TEST_BATCH_SIZE
from mnist.common.mnist_data_loader import MnistDataLoader


def train_mnist_model(model_type, model_path, data_path):
    """
    Trains an mnist model.
    :param model_type: The model type, default: LeNet
    :param model_path: The model path, default: LeNet model path.
    :param data_path: The location of the MNIST data.
    """
    mnist_data_loader = MnistDataLoader(data_path, TEST_BATCH_SIZE)
    model = ModelFactory.create_model(model_type, MNIST_DIMENSIONS, num_classes=NUM_CLASSES)
    mnist_training = ModelTraining(model, mnist_data_loader, training_parameters=TRAINING_PARAMS)
    mnist_training.train()
    mnist_training.save_model(model_path)
    mnist_training.evaluate_plain_text()


def measure_mnist_plain_text_runtime(model_path, data_path, num_runs=20):
    mnist_data_loader = MnistDataLoader(data_path, 1)
    model = torch.load(model_path)
    model_training = ModelTraining(model, mnist_data_loader, training_parameters=TRAINING_PARAMS)
    model_training.measure_plaintext_runtime(num_runs)


def evaluate_plain_text(model_path, data_path):
    mnist_data_loader = MnistDataLoader(data_path, TEST_BATCH_SIZE)
    model = torch.load(model_path)
    model_training = ModelTraining(model, mnist_data_loader, training_parameters=TRAINING_PARAMS)
    model_training.evaluate_plain_text()
