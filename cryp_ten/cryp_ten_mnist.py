import time
import warnings

import crypten
import crypten.mpc as mpc
import torch
from sklearn.metrics import accuracy_score

from common.constants import MNIST_WIDTH, MNIST_HEIGHT, ALICE, BOB
from common.metrics.time_metric import TimeMetric
from common.plain_text_model import PlainTextNet


class CrypTenMnist:

    def __init__(self):
        crypten.init()
        torch.set_num_threads(1)
        warnings.filterwarnings("ignore")

        self.private_model = None
        self.encrypted_data = None
        self.labels = None
        self.encrypted_labels = None

    # Communication is performed using PyTorch distributed backend.
    @mpc.run_multiprocess(world_size=2)
    def encrypt_evaluate_model(self, path_to_model, path_to_data, path_to_labels):
        encrypt_model_metric = TimeMetric("encrypt_model")
        start_time = time.time()
        self.encrypt_model(path_to_model)
        encrypt_model_metric.record(start_time, time.time())

        encrypt_data_metric = TimeMetric("encrypt_data")
        start_time = time.time()
        self.encrypt_data(path_to_data, path_to_labels)
        encrypt_data_metric.record(start_time, time.time())

        evaluate_model_metric = TimeMetric("evaluate_model")
        start_time = time.time()
        self.evaluate_model()
        evaluate_model_metric.record(start_time, time.time())

        encrypt_model_metric.log()
        encrypt_data_metric.log()
        evaluate_model_metric.log()

    def encrypt_model(self, path_to_model):
        dummy_model = PlainTextNet()
        # Note that unlike loading a tensor, the result from crypten.load is not encrypted.
        # Instead, only the src party's model is populated from the file.
        plaintext_model = crypten.load(path_to_model, dummy_model=dummy_model, src=ALICE)
        dummy_input = torch.empty((1, MNIST_WIDTH, MNIST_HEIGHT))

        self.private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
        self.private_model.encrypt(src=ALICE)
        print("Model successfully encrypted:", self.private_model.encrypted)

    def encrypt_data(self, path_to_data, path_to_labels):
        self.encrypted_data = crypten.load(path_to_data, src=BOB)
        self.encrypted_labels = crypten.load(path_to_labels, src=BOB)
        self.labels = torch.load(path_to_labels).long()

    def evaluate_model(self):
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
