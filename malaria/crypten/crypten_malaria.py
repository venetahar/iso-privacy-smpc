import torch
from common.crypten.crypten_private_inference import CryptenPrivateInference
from malaria.common.constants import TEST_BATCH_SIZE
from malaria.common.conv_net import ConvNet
from malaria.common.malaria_training import train_malaria_model, evaluate_plain_text
from malaria.crypten.private_malaria_data_loader import PrivateMalariaDataLoader


def run_crypten_malaria_experiment(model_path, data_path, should_train=False):
    if should_train:
        train_malaria_model(model_path, data_path)

    data_loader = PrivateMalariaDataLoader(data_path)
    evaluate_plain_text(model_path, data_loader)

    dummy_model = ConvNet(in_channels=3, num_classes=2, conv_kernel_sizes=[5, 5], channels=[36, 36],
                          avg_pool_sizes=[2, 2], fc_units=[72])
    dummy_input = torch.empty((1, 3, 32, 32))

    malaria_private_inference = CryptenPrivateInference(dummy_model, dummy_input,
                                                        data_loader, parameters={'test_batch_size': TEST_BATCH_SIZE})
    malaria_private_inference.perform_inference(model_path)
