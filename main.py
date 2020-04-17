from malaria.crypten.crypten_malaria import run_crypten_malaria_experiment
from malaria.pysyft.pysyft_malaria import run_pysyft_malaria_experiment
from mnist.common.constants import MNIST_FC_MODEL_TYPE, MNIST_CONV_MODEL_TYPE
from mnist.cryp_ten.crypten_mnist import run_crypten_mnist_experiment
from mnist.pysyft.pysyft_mnist import run_pysyft_mnist_experiment


MNIST_FC_MODEL_PATH = 'mnist/models/alice_fc_model.pth'
MNIST_CONVNET_MODEL_PATH = 'mnist/models/alice_conv_model.pth'

MALARIA_DATA_PATH = 'malaria/data/'
MALARIA_CONVNET_MODEL_PATH = 'malaria/models/alice_conv_model.pth'


def run_pysyft_experiment(experiment_type, should_retrain_model=False):
    if experiment_type == 'mnist':
        run_pysyft_mnist_experiment(MNIST_FC_MODEL_TYPE, MNIST_FC_MODEL_PATH, should_retrain_model)
        run_pysyft_mnist_experiment(MNIST_CONV_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, should_retrain_model)
    elif experiment_type == 'malaria':
        run_pysyft_malaria_experiment(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH, should_retrain_model)
    else:
        print("Please provide a valid experiment type")


def run_crypten_experiments(experiment_type, should_retrain_model=False):
    if experiment_type == 'mnist':
        run_crypten_mnist_experiment(MNIST_FC_MODEL_TYPE, MNIST_FC_MODEL_PATH, should_retrain_model)
        run_crypten_mnist_experiment(MNIST_CONV_MODEL_TYPE, MNIST_CONVNET_MODEL_PATH, should_retrain_model)
    elif experiment_type == 'malaria':
        run_crypten_malaria_experiment(MALARIA_CONVNET_MODEL_PATH, MALARIA_DATA_PATH, should_retrain_model)
    else:
        print("Please provide a valid experiment type")


run_pysyft_experiment('mnist')
run_crypten_experiments('mnist')

run_pysyft_experiment('malaria')
run_crypten_experiments('malaria')
