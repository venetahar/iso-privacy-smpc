from mnist.common.constants import TEST_BATCH_SIZE
from common.pysyft.pysyft_private_inference import PysyftPrivateInference
from mnist.pysyft.private_pysyft_mnist_data_loader import PrivatePySyftMnistDataLoader


def run_pysyft_mnist_experiment(model_path, data_path):
    private_inference = PysyftPrivateInference(PrivatePySyftMnistDataLoader(data_path),
                                               parameters={'test_batch_size': TEST_BATCH_SIZE})
    private_inference.perform_inference(model_path)
