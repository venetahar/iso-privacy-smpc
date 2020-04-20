from common.model_training.model_training import ModelTraining
from common.models.model_factory import ModelFactory
from mnist.common.constants import TRAINING_PARAMS, MNIST_DIMENSIONS, NUM_CLASSES
from mnist.common.mnist_data_loader import MnistDataLoader


def train_mnist_model(model_type, model_path, data_path):
    """
    Trains an mnist model.
    :param model_type: The model type, default: LeNet
    :param model_path: The model path, default: LeNet model path.
    """
    mnist_data_loader = MnistDataLoader(data_path)
    model = ModelFactory.create_model(model_type, MNIST_DIMENSIONS, num_classes=NUM_CLASSES)
    mnist_training = ModelTraining(model, mnist_data_loader, training_parameters=TRAINING_PARAMS)
    mnist_training.train()
    mnist_training.save_model(model_path)
    mnist_training.evaluate_plain_text()
