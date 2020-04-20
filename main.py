from common.constants import FULLY_CONNECTED3_MODEL_TYPE, CONV_1_MODEL_TYPE
from malaria.common.malaria_training import train_malaria_model
from malaria.crypten.crypten_malaria import run_crypten_malaria_experiment
from malaria.pysyft.pysyft_malaria import run_pysyft_malaria_experiment
from mnist.common.mnist_training import train_mnist_model
from mnist.crypten.crypten_mnist import run_crypten_mnist_experiment, crypten_mnist_benchmark
from mnist.pysyft.pysyft_mnist import run_pysyft_mnist_experiment, pysyft_benchmark_mnist

MNIST_FC_MODEL_PATH = 'mnist/models/alice_fc3_model.pth'
MNIST_CONVNET_MODEL_PATH = 'mnist/models/alice_conv1_model.pth'
MNIST_DATA_PATH = 'mnist/data/'

MALARIA_DATA_PATH = 'malaria/data/'
MALARIA_CONVNET_MODEL_PATH = 'malaria/models/alice_convpool_model.pth'


def run_mnist_fully_connected_experiment(should_retrain_model=False):
    if should_retrain_model:
        train_mnist_model(FULLY_CONNECTED3_MODEL_TYPE, MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    run_crypten_mnist_experiment(FULLY_CONNECTED3_MODEL_TYPE, MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    run_pysyft_mnist_experiment(MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)


def benchmark_mnist_fully_connected_experiment():
    crypten_mnist_benchmark(FULLY_CONNECTED3_MODEL_TYPE, MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    pysyft_benchmark_mnist(MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)


def benchmark_mnist_conv_experiment():
    crypten_mnist_benchmark(CONV_1_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
    pysyft_benchmark_mnist(MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)


def run_mnist_conv_experiment(should_retrain_model=False):
    if should_retrain_model:
        train_mnist_model(CONV_1_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
    run_crypten_mnist_experiment(CONV_1_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
    run_pysyft_mnist_experiment(MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)


def run_malaria_experiment(should_retrain_model=False):
    if should_retrain_model:
        train_malaria_model(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)
    run_crypten_malaria_experiment(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)
    run_pysyft_malaria_experiment(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)


# benchmark_mnist_fully_connected_experiment()
# run_mnist_fully_connected_experiment()
# run_mnist_conv_experiment()
run_malaria_experiment()
