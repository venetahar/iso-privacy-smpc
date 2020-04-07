from common.pysyft.pysyft_private_inference import PysyftPrivateInference
from malaria.common.constants import CONVNET_MODEL_PATH, TEST_BATCH_SIZE
from malaria.common.malaria_training import train_malaria_model, evaluate_plain_text
from malaria.pysyft.private_malaria_data_loader import PrivateMalariaDataLoader

data_path = '../data/cell_images/'
should_train = False
if should_train:
    train_malaria_model(CONVNET_MODEL_PATH, data_path)

data_loader = PrivateMalariaDataLoader(data_path)
evaluate_plain_text(CONVNET_MODEL_PATH, data_loader)

malaria_private_inference = PysyftPrivateInference(data_loader,
                                                   parameters={'test_batch_size': TEST_BATCH_SIZE})
malaria_private_inference.perform_inference(CONVNET_MODEL_PATH)
