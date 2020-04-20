from common.pysyft.pysyft_private_inference import PysyftPrivateInference
from malaria.common.constants import TEST_BATCH_SIZE
from malaria.common.malaria_training import evaluate_plain_text
from malaria.pysyft.private_malaria_data_loader import PrivateMalariaDataLoader


def run_pysyft_malaria_experiment(model_path, data_path):
    data_loader = PrivateMalariaDataLoader(data_path)
    evaluate_plain_text(model_path, data_loader)

    malaria_private_inference = PysyftPrivateInference(data_loader,
                                                       parameters={'test_batch_size': TEST_BATCH_SIZE})
    malaria_private_inference.perform_inference(model_path)
