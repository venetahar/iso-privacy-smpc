from common.constants import FULLY_CONNECTED3_MODEL_TYPE, CONV_1_MODEL_TYPE
from malaria.common.malaria_training import train_malaria_model
from malaria.crypten.crypten_malaria import run_crypten_malaria_experiment, crypten_malaria_benchmark
from malaria.pysyft.pysyft_malaria import run_pysyft_malaria_experiment, pysyft_benchmark_malaria
from mnist.common.mnist_training import train_mnist_model
from mnist.crypten.crypten_mnist import run_crypten_mnist_experiment, crypten_mnist_benchmark
from mnist.pysyft.pysyft_mnist import run_pysyft_mnist_experiment, pysyft_benchmark_mnist

MNIST_FC_MODEL_PATH = 'mnist/models/alice_fc3_model.pth'
MNIST_CONVNET_MODEL_PATH = 'mnist/models/alice_conv1_model.pth'
MNIST_DATA_PATH = 'mnist/data/'

MALARIA_DATA_PATH = 'malaria/data/'
MALARIA_CONVNET_MODEL_PATH = 'malaria/models/alice_convpool_model.pth'


def benchmark_crypten():
    print("====================================================================================")
    print("Benchmarking CrypTen on the MNIST dataset using the Fully Connected model.")
    crypten_mnist_benchmark(FULLY_CONNECTED3_MODEL_TYPE, MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    print("====================================================================================")
    print("Benchmarking CrypTen on the MNIST dataset using the Convolutional model.")
    crypten_mnist_benchmark(CONV_1_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
    print("====================================================================================")
    print("Benchmarking CrypTen on the Malaria dataset using the Convolutional model.")
    crypten_malaria_benchmark(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)


def benchmark_pysyft():
    print("====================================================================================")
    print("Benchmarking PySyft on the MNIST dataset using the Fully Connected model.")
    pysyft_benchmark_mnist(MNIST_FC_MODEL_PATH, MNIST_DATA_PATH)
    print("====================================================================================")
    print("Benchmarking PySyft on the MNIST dataset using the Convolutional model.")
    pysyft_benchmark_mnist(MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH)
    print("====================================================================================")
    print("Benchmarking PySyft on the Malaria dataset using the Convolutional model.")
    pysyft_benchmark_malaria(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH)


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


# run_mnist_conv_experiment('pysyft')
# run_mnist_fully_connected_experiment('pysyft')
run_malaria_experiment('pysyft')

# run_mnist_conv_experiment('crypten')
# run_mnist_fully_connected_experiment('crypten')
# run_malaria_experiment('crypten')

# It is really best to run the benchmarks one at a time as it ensures everything from PySyft and CrypTen is torn down
# properly. There are cases when using PySyft and then CrypTen results in errors probably due to the way they hook
# into pytorch.

# benchmark_crypten()
# benchmark_pysyft()
