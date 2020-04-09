import torch

from mnist.common.conv_plain_text_model import ConvPlainTextNet
from mnist.common.lenet_plain_text_model import PlainTextNet
from mnist.common.mnist_training import train_mnist_model
from mnist.cryp_ten.crypten_private_inference import CryptenPrivateInference
from mnist.common.constants import LENET_MODEL_PATH, CONVNET_MODEL_PATH, MNIST_WIDTH, MNIST_HEIGHT


def evaluate_encrypted(model_type='LeNet', model_path=LENET_MODEL_PATH):
    dummy_input, dummy_model = get_dummy_values(model_type)
    crypten_model = CryptenPrivateInference(dummy_model, dummy_input, './data/bob_test.pth',
                                            './data/bob_test_labels.pth')
    crypten_model.perform_inference(model_path)


def get_dummy_values(model_type):
    if model_type == 'ConvNet':
        dummy_model = ConvPlainTextNet()
        dummy_input = torch.empty((1, 1, MNIST_WIDTH, MNIST_HEIGHT))

    else:
        dummy_model = PlainTextNet()
        dummy_input = torch.empty((1, 1, MNIST_WIDTH, MNIST_HEIGHT))
    return dummy_input, dummy_model


should_train = True
if should_train:
    train_mnist_model('ConvNet', CONVNET_MODEL_PATH)
evaluate_encrypted('ConvNet', CONVNET_MODEL_PATH)
# evaluate_encrypted('LeNet', LENET_MODEL_PATH)
