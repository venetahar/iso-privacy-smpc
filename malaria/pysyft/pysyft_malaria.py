from common.pysyft.pysyft_private_inference import PysyftPrivateInference
from malaria.common.constants import TEST_BATCH_SIZE
from malaria.common.malaria_training import evaluate_saved_model
from malaria.pysyft.private_malaria_data_loader import PrivateMalariaDataLoader


def run_pysyft_malaria_experiment(model_path, data_path):
    data_loader = PrivateMalariaDataLoader(data_path, TEST_BATCH_SIZE)
    evaluate_saved_model(model_path, data_loader)

    malaria_private_inference = PysyftPrivateInference(data_loader,
                                                       parameters={'test_batch_size': TEST_BATCH_SIZE})
    malaria_private_inference.perform_inference(model_path)


def pysyft_benchmark_malaria(model_path, data_path):
    test_batch_size = 1
    data_loader = PrivateMalariaDataLoader(data_path, test_batch_size)

    malaria_private_inference = PysyftPrivateInference(data_loader,
                                                       parameters={'test_batch_size': test_batch_size})
    malaria_private_inference.measure_communication_costs(model_path)

    # Best to create a new object as extra logging is enabled which could introduce slowdowns
    malaria_private_inference = PysyftPrivateInference(data_loader,
                                                       parameters={'test_batch_size': test_batch_size})
    malaria_private_inference.measure_runtime(model_path)
