import warnings

import crypten
import crypten.mpc as mpc
import torch
from sklearn.metrics import accuracy_score

from common.crypten.constants import ALICE, BOB
from common.private_inference import PrivateInference


class CryptenPrivateInference(PrivateInference):
    """
    Class encapsulating the logic for performing private inference using Crypten.
    """

    def __init__(self, dummy_model, dummy_input, path_to_data, path_to_labels):
        """
        Creates a CryptenPrivateInference.
        :param dummy_model: The dummy model, required by Crypten.
        :param dummy_input: The dummy input, required by Crypten.
        :param path_to_data: The path to the data.
        :param path_to_labels: The path to the labels.
        """
        crypten.init()
        torch.set_num_threads(1)
        warnings.filterwarnings("ignore")

        self.dummy_model = dummy_model
        self.dummy_input = dummy_input
        self.path_to_data = path_to_data
        self.path_to_labels = path_to_labels

        self.private_model = None
        self.encrypted_data = None
        self.labels = None
        self.encrypted_labels = None

    # Communication is performed using PyTorch distributed backend.
    @mpc.run_multiprocess(world_size=2)
    def perform_inference(self, path_to_model):
        super().perform_inference(path_to_model)

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
        self.encrypted_data = crypten.load(self.path_to_data, src=BOB)
        self.encrypted_labels = crypten.load(self.path_to_labels, src=BOB)
        self.labels = torch.load(self.path_to_labels).long()

    def evaluate(self):
        """
        Performs secure evaluation of the model.
        """
        self.private_model.eval()
        output_enc = self.private_model(self.encrypted_data)
        # Weirdly these produce different results so for now we have to use the decrypted values
        # correct = output_enc.argmax(dim=1).eq(self.encrypted_labels).sum()
        # print(correct.get_plain_text())
        # print(output_enc.argmax(dim=1).get_plain_text().eq(self.encrypted_labels.get_plain_text()).sum())
        output = output_enc.get_plain_text()
        predictions = torch.max(output.data, 1)[1]
        accuracy = accuracy_score(predictions, self.labels)
        print("\tAccuracy: {0:.4f}".format(accuracy))
