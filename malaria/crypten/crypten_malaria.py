import torch

from common.constants import CONVPOOL_MODEL_TYPE
from common.crypten.crypten_private_inference import CryptenPrivateInference
from common.model_factory import ModelFactory
from malaria.common.constants import TEST_BATCH_SIZE
from malaria.common.malaria_training import evaluate_saved_model
from malaria.crypten.private_malaria_data_loader import PrivateMalariaDataLoader


def run_crypten_malaria_experiment(model_path, data_path):
    malaria_private_inference = _get_crypten_private_inference(model_path, data_path, TEST_BATCH_SIZE)
    malaria_private_inference.perform_inference(model_path)


def crypten_malaria_benchmark(model_path, data_path):
    crypten_private_inference = _get_crypten_private_inference(model_path, data_path, 1)
    crypten_private_inference.measure_runtime(model_path)
    crypten_private_inference.measure_communication_costs(model_path)


def _get_crypten_private_inference(model_path, data_path, test_batch_size):
    data_loader = PrivateMalariaDataLoader(data_path, test_batch_size)
    evaluate_saved_model(model_path, data_loader)

    dummy_model = ModelFactory.create_model(CONVPOOL_MODEL_TYPE, (32, 32, 3), num_classes=2)
    dummy_input = torch.empty((1, 3, 32, 32))

    return CryptenPrivateInference(dummy_model, dummy_input, data_loader)
