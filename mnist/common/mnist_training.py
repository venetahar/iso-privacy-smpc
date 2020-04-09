from common.model_training.model_training import ModelTraining
from common.utils.data_utils import DataUtils
from mnist.common.constants import FC_MODEL_PATH, TRAINING_PARAMS, CONV_MODEL_TYPE, FC_MODEL_TYPE
from mnist.common.conv_model import ConvModel
from mnist.common.fully_connected_model import FullyConnectedModel
from mnist.common.mnist_data_loader import MnistDataLoader


def train_mnist_model(model_type=FC_MODEL_TYPE, model_path=FC_MODEL_PATH):
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
    DataUtils.save_data('./data/bob', mnist_data_loader.test_set)


def get_model(model_type):
    """
    Returns the model based on the model type.
    :param model_type: The model type.
    :return: The specified model.
    """
    if model_type == CONV_MODEL_TYPE:
        model = ConvModel()
    else:
        model = FullyConnectedModel()
    return model
