import argparse

from common.constants import FULLY_CONNECTED3_MODEL_TYPE, CONV_1_MODEL_TYPE
from main import MALARIA_CONVNET_MODEL_PATH, MNIST_FC_MODEL_PATH, MNIST_CONVNET_MODEL_PATH, MALARIA_DATA_PATH, \
    MNIST_DATA_PATH
from malaria.crypten.crypten_malaria import crypten_malaria_benchmark
from malaria.pysyft.pysyft_malaria import pysyft_benchmark_malaria
from mnist.crypten.crypten_mnist import crypten_mnist_benchmark
from mnist.pysyft.pysyft_mnist import pysyft_benchmark_mnist


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


def benchmark_pysyft(protocol='snn'):
    print("====================================================================================")
    print("Benchmarking PySyft on the MNIST dataset using the Fully Connected model.")
    pysyft_benchmark_mnist(MNIST_FC_MODEL_PATH, MNIST_DATA_PATH, protocol=protocol)
    print("====================================================================================")
    print("Benchmarking PySyft on the MNIST dataset using the Convolutional model.")
    pysyft_benchmark_mnist(MNIST_CONVNET_MODEL_PATH, MNIST_DATA_PATH, protocol=protocol)
    print("====================================================================================")
    print("Benchmarking PySyft on the Malaria dataset using the Convolutional model.")
    pysyft_benchmark_malaria(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH, protocol=protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, default='',
                        help='The framework name. Can be either: pysyft or crypten')
    parser.add_argument('--pysyft_protocol', type=str, default='snn',
                        help='pysyft protocol. Can be either: snn or fss. Default: snn.')
    config = parser.parse_args()

    if config.framework == 'crypten':
        benchmark_crypten()
    elif config.framework == 'pysyft':
        benchmark_pysyft(config.pysyft_protocol)
    else:
        print("Please supply a valid framework type. Can be either: crypten or pysyft.")
