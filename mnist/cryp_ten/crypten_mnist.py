import torch

from mnist.common.conv_model import ConvModel
from mnist.common.fully_connected_model import FullyConnectedModel
from mnist.common.mnist_training import train_mnist_model
from common.crypten.crypten_private_inference import CryptenPrivateInference
from mnist.common.constants import FC_MODEL_PATH, CONVNET_MODEL_PATH, MNIST_WIDTH, MNIST_HEIGHT, CONV_MODEL_TYPE, \
    FC_MODEL_TYPE, TEST_BATCH_SIZE
from mnist.cryp_ten.private_mnist_data_loader import PrivateMnistDataLoader


def evaluate_encrypted(model_type=FC_MODEL_TYPE, model_path=FC_MODEL_PATH):
    dummy_input, dummy_model = get_dummy_values(model_type)
    crypten_model = CryptenPrivateInference(dummy_model, dummy_input, PrivateMnistDataLoader(),
                                            parameters={'test_batch_size': TEST_BATCH_SIZE})
    crypten_model.perform_inference(model_path)


def get_dummy_values(model_type):
    if model_type == CONV_MODEL_TYPE:
        dummy_model = ConvModel()
        dummy_input = torch.empty((1, 1, MNIST_WIDTH, MNIST_HEIGHT))

    else:
        dummy_model = FullyConnectedModel()
        dummy_input = torch.empty((1, 1, MNIST_WIDTH, MNIST_HEIGHT))
    return dummy_input, dummy_model


should_train = False
if should_train:
    train_mnist_model(FC_MODEL_TYPE, FC_MODEL_PATH)
evaluate_encrypted(FC_MODEL_TYPE, FC_MODEL_PATH)
