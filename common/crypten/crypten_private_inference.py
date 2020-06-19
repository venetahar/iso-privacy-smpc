import warnings

import crypten
import logging
import crypten.mpc as mpc
import torch
from crypten import cryptensor

from common.constants import ALICE, BOB
from common.private_inference import PrivateInference


class CryptenPrivateInference(PrivateInference):
    """
    Class encapsulating the logic for performing private inference using Crypten.
    """

    def __init__(self, dummy_model, dummy_input, test_data_loader, parameters=None):
        """
        Creates a CryptenPrivateInference.
        :param dummy_model: The dummy model, required by Crypten.
        :param dummy_input: The dummy input, required by Crypten.
        :param test_data_loader: The data_loader.
        :param parameters: The parameters.
        """
        super(CryptenPrivateInference, self).__init__(test_data_loader, parameters)
        crypten.init()
        warnings.filterwarnings("ignore")

        self.dummy_model = dummy_model
        self.dummy_input = dummy_input

    # Communication is performed using PyTorch distributed backend.
    @mpc.run_multiprocess(world_size=2)
    def perform_inference(self, path_to_model):
        """
        Performs inference using a stored model.
        :param path_to_model: The model path.
        """
        # Set the verbosity of the communicator to True, needed to ensure that costs will be logged.
        super().perform_inference(path_to_model)

    @mpc.run_multiprocess(world_size=2)
    def measure_runtime(self, path_to_model):
        """
        Measures the runtime of performing private inference.
        :param path_to_model: The model path.
        """
        super().measure_runtime(path_to_model)

    @mpc.run_multiprocess(world_size=2)
    def measure_communication_costs(self, path_to_model):
        """
        Measures the communication costs of performing private inference.
        :param path_to_model: The model path.
        """
        # Set the logging level to INFO to be able to display the communication costs
        level = logging.INFO
        logging.getLogger().setLevel(level)
        torch.set_num_threads(1)
        crypten.comm.get().set_verbosity(True)

        super().measure_communication_costs(path_to_model)

        crypten.print_communication_stats()

    def encrypt_model(self, path_to_model):
        """
        Encrypts the model.
        :param path_to_model: The path to the saved model to be loaded and secret shared.
        """
        # Note that unlike loading a tensor, the result from crypten.load is not encrypted.
        # Instead, only the src party's model is populated from the file.
        plaintext_model = crypten.load(path_to_model, dummy_model=self.dummy_model, src=ALICE)

        self.private_model = crypten.nn.from_pytorch(plaintext_model, self.dummy_input)
        self.private_model.encrypt(src=ALICE)

    def encrypt_data(self):
        """
        Encrypts the data.
        """
        self.test_data_loader.encrypt_test_data(BOB)

    def encrypt_single_instance(self, data_instance):
        """
        Encrypts a single data instance.
        :param data_instance: The data instance.
        :return: The encrypted / secret shared data instance.
        """
        return cryptensor(data_instance, src=BOB)

    def evaluate(self):
        """
        Performs secure evaluation of the model.
        """
        self.private_model.eval()
        correct_predictions = 0
        total_predictions = 0
        for batch_index, (data, target) in enumerate(self.test_data_loader.private_test_loader):
            print('Performing inference for batch {}'.format(batch_index))
            output_enc = self.private_model(data)
            # Weirdly these produce different results so for now we have to use the decrypted values
            # correct = output_enc.argmax(dim=1).eq(self.encrypted_labels).sum()
            # print(correct.get_plain_text())
            correct_predictions += output_enc.argmax(dim=1).get_plain_text().eq(target.get_plain_text()).sum()
            total_predictions += len(target)

        accuracy = 100.0 * correct_predictions / total_predictions
        print('Crypten Test set: Accuracy: {}/{} ({:.4f}%)'.format(correct_predictions, total_predictions, accuracy))
