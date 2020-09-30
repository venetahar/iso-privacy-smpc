# THIS IS CRITICAL FOR CRYPTEN TO WORK ON CERTAIN LINUX DISTRIBUTIONS!
# For more details see: https://github.com/facebookresearch/CrypTen/issues/88
import argparse

import torch

torch.set_num_threads(1)

from numpy.random import seed
seed(0)

torch.manual_seed(0)

from common.constants import FULLY_CONNECTED3_MODEL_TYPE, CONV_1_MODEL_TYPE
from malaria.common.malaria_training import train_malaria_model
from malaria.crypten.crypten_malaria import run_crypten_malaria_experiment
from malaria.pysyft.pysyft_malaria import run_pysyft_malaria_experiment
from mnist.common.mnist_training import train_mnist_model
from mnist.crypten.crypten_mnist import run_crypten_mnist_experiment
from mnist.pysyft.pysyft_mnist import run_pysyft_mnist_experiment

MNIST_FC_MODEL_PATH = 'mnist/models/alice_fc3_model.pth'
MNIST_CONVNET_MODEL_PATH = 'mnist/models/alice_conv1_model.pth'
MNIST_DATA_PATH = 'mnist/data/'

MALARIA_DATA_PATH = 'malaria/data/'
MALARIA_CONVNET_MODEL_PATH = 'malaria/models/alice_convpool_model.pth'


def run_mnist_fully_connected_experiment(framework, should_retrain_model=False):
    if should_retrain_model:
        train_mnist_model(FULLY_CONNECTED3_MODEL_TYPE, MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    if framework == 'crypten':
        run_crypten_mnist_experiment(FULLY_CONNECTED3_MODEL_TYPE, MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    else:
        run_pysyft_mnist_experiment(MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)


def run_mnist_conv_experiment(framework, should_retrain_model=False):
    if should_retrain_model:
        train_mnist_model(CONV_1_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
    if framework == 'crypten':
        run_crypten_mnist_experiment(CONV_1_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
    else:
        run_pysyft_mnist_experiment(MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)


def run_malaria_experiment(framework, should_retrain_model=False):
    if should_retrain_model:
        train_malaria_model(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)
    if framework == 'crypten':
        run_crypten_malaria_experiment(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)
    else:
        run_pysyft_malaria_experiment(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='',
                        help='The experiment name. Can be either: mnist_fc, mnist_conv or malaria_conv')
    parser.add_argument('--framework', type=str, default='',
                        help='The framework name. Can be either: pysyft or crypten')
    parser.add_argument('--retrain', default=False, action="store_true",
                        help='Whether to retrain the model. Default False.')
    config = parser.parse_args()

    if config.experiment_name == 'mnist_fc':
        run_mnist_fully_connected_experiment(config.framework, config.retrain)
    elif config.experiment_name == 'mnist_conv':
        run_mnist_conv_experiment(config.framework, config.retrain)
    elif config.experiment_name == 'malaria_conv':
        run_malaria_experiment(config.framework, config.retrain)
    else:
        print("Please supply a valid experiment type. Can be either: mnist_fc, mnist_conv or malaria_conv ")