from common.graphene.rest.graphene_rest_private_inference import GrapheneRESTClientPrivateInference
from mnist.common.constants import TEST_BATCH_SIZE
from mnist.common.mnist_training import evaluate_plain_text
from mnist.common.mnist_data_loader import MnistDataLoader
from common.utils.python_utils import get_model_type
from common.constants import MODELS


def run_graphene_rest_mnist_experiment(model_path, data_path):

    model_type = get_model_type(model_path, MODELS)
    evaluate_plain_text(model_path, data_path)
    private_inference = GrapheneRESTClientPrivateInference("http://127.0.0.1:5000", MnistDataLoader(data_path, TEST_BATCH_SIZE))
    private_inference.perform_inference(model_type)


def graphene_rest_benchmark_mnist(model_path, data_path):

    model_type = get_model_type(model_path, MODELS)
    # TODO not implemented
    # private_inference = GrapheneRESTClientPrivateInference("http://127.0.0.1:5000", MnistDataLoader(data_path, 1))
    # private_inference.measure_communication_costs(model_path)

    # Best to create a new object as extra logging is enabled which could introduce slowdowns
    private_inference = GrapheneRESTClientPrivateInference("http://127.0.0.1:5000", MnistDataLoader(data_path, 1))
    private_inference.measure_runtime(model_type)
