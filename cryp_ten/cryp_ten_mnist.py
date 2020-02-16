import warnings

import crypten
import crypten.mpc as mpc
import torch
from sklearn.metrics import accuracy_score

from cryp_ten.constants import NUM_SAMPLES, MNIST_WIDTH, MNIST_HEIGHT, ALICE, BOB
from cryp_ten.plain_text_model import PlainTextNet


class CrypTenMnist:

    def __init__(self):
        crypten.init()
        torch.set_num_threads(1)
        warnings.filterwarnings("ignore")

        self.private_model = None
        self.encrypted_data = None
        self.labels = None

    @mpc.run_multiprocess(world_size=2)
    def encrypt_evaluate_model(self, path_to_model, path_to_data, path_to_labels):
        self.encrypt_model(path_to_model)
        self.encrypt_data(path_to_data, path_to_labels)
        self.evaluate_model()

    def encrypt_model(self, path_to_model):
        dummy_model = PlainTextNet()
        plaintext_model = crypten.load(path_to_model, dummy_model=dummy_model, src=ALICE)
        dummy_input = torch.empty((1, MNIST_WIDTH, MNIST_HEIGHT))

        self.private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
        self.private_model.encrypt(src=ALICE)
        print("Model successfully encrypted:", self.private_model.encrypted)

    def encrypt_data(self, path_to_data, path_to_labels):
        data_enc = crypten.load(path_to_data, src=BOB)
        self.encrypted_data = data_enc[:NUM_SAMPLES]
        self.labels = torch.load(path_to_labels).long()

    def evaluate_model(self):
        self.private_model.eval()
        output_enc = self.private_model(self.encrypted_data)
        output = output_enc.get_plain_text()
        predictions = torch.max(output.data, 1)[1]
        accuracy = accuracy_score(predictions, self.labels[:NUM_SAMPLES])
        print("\tAccuracy: {0:.4f}".format(accuracy))
