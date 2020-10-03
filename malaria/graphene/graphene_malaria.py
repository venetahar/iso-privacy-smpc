from common.graphene.rest.graphene_rest_private_inference import GrapheneRESTClientPrivateInference
from malaria.common.constants import TEST_BATCH_SIZE
from malaria.common.malaria_training import evaluate_saved_model
from common.utils.python_utils import get_model_type
from common.constants import MODELS
from malaria.graphene.json_graphene_rest_malaria_data_loader import JSONGrapheneRESTMalariaDataLoader


def run_graphene_rest_malaria_experiment(model_path, data_path):

    data_loader = JSONGrapheneRESTMalariaDataLoader(data_path, TEST_BATCH_SIZE)

    model_type = get_model_type(model_path, MODELS)
    evaluate_saved_model(model_path, data_loader)
    private_inference = GrapheneRESTClientPrivateInference("http://127.0.0.1:5000", data_loader)
    private_inference.perform_inference(model_type)


def graphene_rest_benchmark_malaria(model_path, data_path):

    model_type = get_model_type(model_path, MODELS)
    # TODO not implemented
    # private_inference = GrapheneRESTClientPrivateInference("http://127.0.0.1:5000", MnistDataLoader(data_path, 1))
    # private_inference.measure_communication_costs(model_path)

    data_loader = JSONGrapheneRESTMalariaDataLoader(data_path, 1)

    # Best to create a new object as extra logging is enabled which could introduce slowdowns
    private_inference = GrapheneRESTClientPrivateInference("http://127.0.0.1:5000", data_loader)
    private_inference.measure_runtime(model_type, num_runs=100)
