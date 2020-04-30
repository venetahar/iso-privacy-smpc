from common.pysyft.pysyft_private_inference import PysyftPrivateInference
from mnist.common.constants import TEST_BATCH_SIZE
from mnist.common.mnist_training import evaluate_plain_text
from mnist.pysyft.private_pysyft_mnist_data_loader import PrivatePySyftMnistDataLoader


def run_pysyft_mnist_experiment(model_path, data_path):
    evaluate_plain_text(model_path, data_path)
    private_inference = PysyftPrivateInference(PrivatePySyftMnistDataLoader(data_path, TEST_BATCH_SIZE))
    private_inference.perform_inference(model_path)


def pysyft_benchmark_mnist(model_path, data_path):
    private_inference = PysyftPrivateInference(PrivatePySyftMnistDataLoader(data_path, 1))
    private_inference.measure_communication_costs(model_path)

    # Best to create a new object as extra logging is enabled which could introduce slowdowns
    private_inference = PysyftPrivateInference(PrivatePySyftMnistDataLoader(data_path, 1))
    private_inference.measure_runtime(model_path)
