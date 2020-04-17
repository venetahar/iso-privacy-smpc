from common.model_training.model_training import ModelTraining
from mnist.common.constants import TRAINING_PARAMS, MNIST_CONV_MODEL_TYPE
from mnist.common.conv_model import ConvModel
from mnist.common.fully_connected_model import FullyConnectedModel
from mnist.common.mnist_data_loader import MnistDataLoader


def train_mnist_model(model_type, model_path):
    """
    Trains an mnist model.
    :param model_type: The model type, default: LeNet
    :param model_path: The model path, default: LeNet model path.
    """
    mnist_data_loader = MnistDataLoader()
    model = get_model(model_type)
    mnist_training = ModelTraining(model, mnist_data_loader, training_parameters=TRAINING_PARAMS)
    mnist_training.train()
    mnist_training.save_model(model_path)
    mnist_training.evaluate_plain_text()


def get_model(model_type):
    """
    Returns the model based on the model type.
    :param model_type: The model type.
    :return: The specified model.
    """
    if model_type == MNIST_CONV_MODEL_TYPE:
        model = ConvModel()
    else:
        model = FullyConnectedModel()
    return model
