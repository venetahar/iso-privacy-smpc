import torch

from common.crypten.crypten_private_inference import CryptenPrivateInference
from common.models.model_factory import ModelFactory
from mnist.common.constants import TEST_BATCH_SIZE, MNIST_DIMENSIONS, \
    NUM_CLASSES
from mnist.crypten.private_crypten_mnist_data_loader import PrivateCryptenMnistDataLoader


def run_crypten_mnist_experiment(model_type, model_path, data_path):
    dummy_model = ModelFactory.create_model(model_type, MNIST_DIMENSIONS, NUM_CLASSES)
    image_width, image_height, in_channels = MNIST_DIMENSIONS
    dummy_input = torch.empty((1, in_channels, image_width, image_height))
    crypten_model = CryptenPrivateInference(dummy_model, dummy_input, PrivateCryptenMnistDataLoader(data_path),
                                            parameters={'test_batch_size': TEST_BATCH_SIZE})
    crypten_model.perform_inference(model_path)
