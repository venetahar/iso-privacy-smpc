import torch

from mnist.common.conv_model import ConvModel
from mnist.common.fully_connected_model import FullyConnectedModel
from mnist.common.mnist_training import train_mnist_model
from common.crypten.crypten_private_inference import CryptenPrivateInference
from mnist.common.constants import MNIST_WIDTH, MNIST_HEIGHT, TEST_BATCH_SIZE, MNIST_CONV_MODEL_TYPE
from mnist.cryp_ten.private_mnist_data_loader import PrivateMnistDataLoader


def _evaluate_encrypted(model_type, model_path):
    dummy_input, dummy_model = _get_dummy_values(model_type)
    crypten_model = CryptenPrivateInference(dummy_model, dummy_input, PrivateMnistDataLoader(),
                                            parameters={'test_batch_size': TEST_BATCH_SIZE})
    crypten_model.perform_inference(model_path)


def _get_dummy_values(model_type):
    if model_type == MNIST_CONV_MODEL_TYPE:
        dummy_model = ConvModel()
        dummy_input = torch.empty((1, 1, MNIST_WIDTH, MNIST_HEIGHT))

    else:
        dummy_model = FullyConnectedModel()
        dummy_input = torch.empty((1, 1, MNIST_WIDTH, MNIST_HEIGHT))
    return dummy_input, dummy_model


def run_crypten_mnist_experiment(model_type, model_path, should_train=False):
    if should_train:
        train_mnist_model(model_type, model_path)
    _evaluate_encrypted(model_type, model_path)
