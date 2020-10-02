# THIS IS CRITICAL FOR CRYPTEN TO WORK ON CERTAIN LINUX DISTRIBUTIONS!
# For more details see: https://github.com/facebookresearch/CrypTen/issues/88
import argparse

import torch
torch.set_num_threads(1)

from numpy.random import seed
seed(0)

torch.manual_seed(0)

from common.constants import FULLY_CONNECTED3_MODEL_TYPE, CONV_1_MODEL_TYPE
from malaria.common.malaria_training import train_malaria_model, measure_malaria_plain_text_runtime
from malaria.crypten.crypten_malaria import run_crypten_malaria_experiment, crypten_malaria_benchmark
from malaria.pysyft.pysyft_malaria import run_pysyft_malaria_experiment, pysyft_benchmark_malaria
from mnist.common.mnist_training import train_mnist_model, measure_mnist_plain_text_runtime
from mnist.crypten.crypten_mnist import run_crypten_mnist_experiment, crypten_mnist_benchmark
from mnist.pysyft.pysyft_mnist import run_pysyft_mnist_experiment, pysyft_benchmark_mnist
from mnist.graphene.graphene_mnist import run_graphene_rest_mnist_experiment, graphene_rest_benchmark_mnist

MNIST_FC_MODEL_PATH = 'mnist/models/alice_fc3_model.pth'
MNIST_CONVNET_MODEL_PATH = 'mnist/models/alice_conv1_model.pth'
MNIST_DATA_PATH = 'mnist/data/'

MALARIA_DATA_PATH = 'malaria/data/'
MALARIA_CONVNET_MODEL_PATH = 'malaria/models/alice_convpool_model.pth'


def run_mnist_fully_connected_experiment(framework, should_retrain_model=False, should_benchmark=False):
    if should_retrain_model:
        train_mnist_model(FULLY_CONNECTED3_MODEL_TYPE, MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    if framework == 'crypten':
        if should_benchmark:
            print("====================================================================================")
            print("Benchmarking CrypTen on the MNIST dataset using the Fully Connected model.")
            crypten_mnist_benchmark(FULLY_CONNECTED3_MODEL_TYPE, MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
        else:
            run_crypten_mnist_experiment(FULLY_CONNECTED3_MODEL_TYPE, MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    elif framework == 'pysyft':
        if should_benchmark:
            print("====================================================================================")
            print("Benchmarking PySyft on the MNIST dataset using the Fully Connected model.")
            pysyft_benchmark_mnist(MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
        else:
            run_pysyft_mnist_experiment(MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    elif framework == 'graphene':
        if should_benchmark:
            print("====================================================================================")
            print("Benchmarking Graphene on the MNIST dataset using the Fully Connected model.")
            graphene_rest_benchmark_mnist(MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
        else:
            run_graphene_rest_mnist_experiment(MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    else:
        print("Please supply a valid framework. Can be either: pysyft, crypten or graphene ")


def run_mnist_conv_experiment(framework, should_retrain_model=False, should_benchmark=False):
    if should_retrain_model:
        train_mnist_model(CONV_1_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
    if framework == 'crypten':
        if should_benchmark:
            print("====================================================================================")
            print("Benchmarking CrypTen on the Malaria dataset using the Convolutional model.")
            crypten_malaria_benchmark(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)
        else:
            run_crypten_mnist_experiment(CONV_1_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
    else:
        if should_benchmark:
            print("====================================================================================")
            print("Benchmarking PySyft on the MNIST dataset using the Convolutional model.")
            pysyft_benchmark_mnist(MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
        else:
            run_pysyft_mnist_experiment(MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)


def run_malaria_experiment(framework, should_retrain_model=False, should_benchmark=False):
    if should_retrain_model:
        train_malaria_model(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)
    if framework == 'crypten':
        if should_benchmark:
            print("====================================================================================")
            print("Benchmarking CrypTen on the Malaria dataset using the Convolutional model.")
            crypten_malaria_benchmark(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)
        else:
            run_crypten_malaria_experiment(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)
    else:
        if should_benchmark:
            print("====================================================================================")
            print("Benchmarking PySyft on the Malaria dataset using the Convolutional model.")
            pysyft_benchmark_malaria(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)
        else:
            run_pysyft_malaria_experiment(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='',
                        help='The experiment name. Can be either: mnist_fc, mnist_conv or malaria_conv')
    parser.add_argument('--framework', type=str, default='',
                        help='The framework name. Can be either: pysyft, crypten or graphene')
    parser.add_argument('--benchmark', default=False, action="store_true",
                        help='Whether to benchmark the experiment. Default False.')
    parser.add_argument('--retrain', default=False, action="store_true",
                        help='Whether to retrain the model. Default False.')
    parser.add_argument('--pysyft_protocol', type=str, default='snn',
                        help='pysyft protocol. Can be either: snn or fss. Default: snn.')
    config = parser.parse_args()

    if config.experiment_name == 'mnist_fc':
        run_mnist_fully_connected_experiment(config.framework, config.retrain, config.benchmark)
    elif config.experiment_name == 'mnist_conv':
        run_mnist_conv_experiment(config.framework, config.retrain, config.benchmark)
    elif config.experiment_name == 'malaria_conv':
        run_malaria_experiment(config.framework, config.retrain, config.benchmark)
    else:
        print("Please supply a valid experiment type. Can be either: mnist_fc, mnist_conv or malaria_conv ")