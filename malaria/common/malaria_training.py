from common.model_training.model_training import ModelTraining
from malaria.common.constants import TRAINING_PARAMS
from malaria.common.conv_net import ConvNet
from malaria.common.malaria_data_loader import MalariaDataLoader


def train_malaria_model(model_path, data_path):
    data_loader = MalariaDataLoader(data_path, should_load_split=False)
    model = ConvNet(in_channels=3, num_classes=2, conv_kernel_sizes=[5, 5], channels=[36, 36], max_pool_sizes=[2, 2],
                    fc_units=[72])
    malaria_training = ModelTraining(model, data_loader, training_parameters=TRAINING_PARAMS)
    malaria_training.train()
    malaria_training.evaluate_plain_text()
    malaria_training.save_model(model_path)
