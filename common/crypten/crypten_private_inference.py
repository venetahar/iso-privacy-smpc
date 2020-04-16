import warnings

import crypten
import logging
import crypten.mpc as mpc
import torch

from common.crypten.constants import ALICE, BOB
from common.private_inference import PrivateInference


class CryptenPrivateInference(PrivateInference):
    """
    Class encapsulating the logic for performing private inference using Crypten.
    """

    def __init__(self, dummy_model, dummy_input, data_loader, parameters):
        """
        Creates a CryptenPrivateInference.
        :param dummy_model: The dummy model, required by Crypten.
        :param dummy_input: The dummy input, required by Crypten.
        :param data_loader: The data_loader.
        :param parameters: The parameters.
        """
        crypten.init()
        # Set the logging level to INFO to be able to display the communication costs
        level = logging.INFO
        logging.getLogger().setLevel(level)
        torch.set_num_threads(1)
        warnings.filterwarnings("ignore")

        self.dummy_model = dummy_model
        self.dummy_input = dummy_input
        self.data_loader = data_loader
        self.parameters = parameters

        self.private_model = None

    # Communication is performed using PyTorch distributed backend.
    @mpc.run_multiprocess(world_size=2)
    def perform_inference(self, path_to_model):
        # Set the verbosity of the communicator to True, needed to ensure that costs will be logged.
        crypten.comm.get().set_verbosity(True)
        super().perform_inference(path_to_model)

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
        print("Model successfully encrypted:", self.private_model.encrypted)

    def encrypt_data(self):
        """
        Encrypts the data.
        """
        self.data_loader.encrypt_test_data(BOB)

    def evaluate(self):
        """
        Performs secure evaluation of the model.
        """
        self.private_model.eval()
        correct_predictions = 0
        total_predictions = len(self.data_loader.private_test_loader) * self.parameters['test_batch_size']
        for batch_index, (data, target) in enumerate(self.data_loader.private_test_loader):
            output_enc = self.private_model(data)
            # Weirdly these produce different results so for now we have to use the decrypted values
            # correct = output_enc.argmax(dim=1).eq(self.encrypted_labels).sum()
            # print(correct.get_plain_text())
            correct_predictions += output_enc.argmax(dim=1).get_plain_text().eq(target.get_plain_text()).sum()

        accuracy = 100.0 * correct_predictions / total_predictions
        print('Test set: Accuracy: {}/{} ({:.4f}%)'.format(correct_predictions, total_predictions, accuracy))
