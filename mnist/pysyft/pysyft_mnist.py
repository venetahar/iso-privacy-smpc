from mnist.common.constants import TEST_BATCH_SIZE
from mnist.common.mnist_training import train_mnist_model
from common.pysyft.pysyft_private_inference import PysyftPrivateInference
from mnist.pysyft.private_mnist_data_loader import PrivateMnistDataLoader


def run_pysyft_mnist_experiment(model_type, model_path, should_train=False):
    if should_train:
        train_mnist_model(model_type, model_path)

    smpc_mnist = PysyftPrivateInference(PrivateMnistDataLoader(), parameters={'test_batch_size': TEST_BATCH_SIZE})
    smpc_mnist.perform_inference(model_path)
