import torch

from common.crypten.crypten_private_inference import CryptenPrivateInference
from common.model_factory import ModelFactory
from mnist.common.constants import TEST_BATCH_SIZE, MNIST_DIMENSIONS, NUM_CLASSES
from mnist.common.mnist_training import evaluate_plain_text
from mnist.crypten.private_crypten_mnist_data_loader import PrivateCryptenMnistDataLoader


def run_crypten_mnist_experiment(model_type, model_path, data_path):
    evaluate_plain_text(model_path, data_path)
    crypten_private_inference = get_crypten_private_inference(data_path, model_type, TEST_BATCH_SIZE)
    crypten_private_inference.perform_inference(model_path)


def crypten_mnist_benchmark(model_type, model_path, data_path):
    crypten_private_inference = get_crypten_private_inference(data_path, model_type, 1)
    crypten_private_inference.measure_runtime(model_path)
    crypten_private_inference.measure_communication_costs(model_path)


def get_crypten_private_inference(data_path, model_type, test_batch_size):
    dummy_model = ModelFactory.create_model(model_type, MNIST_DIMENSIONS, NUM_CLASSES)
    image_width, image_height, in_channels = MNIST_DIMENSIONS
    dummy_input = torch.empty((1, in_channels, image_width, image_height))
    return CryptenPrivateInference(dummy_model, dummy_input, PrivateCryptenMnistDataLoader(data_path, test_batch_size))
