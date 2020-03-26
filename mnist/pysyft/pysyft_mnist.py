import time

import syft as sy
import torch

from mnist.common import TEST_BATCH_SIZE
from mnist.common import TimeMetric
from mnist.pysyft import PrivateMnistDataLoader


class PysyftMnist:

    def __init__(self):
        hook = sy.TorchHook(torch)
        self.client = sy.VirtualWorker(hook, id="client")
        self.bob = sy.VirtualWorker(hook, id="bob")
        self.alice = sy.VirtualWorker(hook, id="alice")
        self.crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
        self.data_loader = None
        self.model = None

    def encrypt_evaluate_model(self, path_to_model):
        encrypt_model_metric = TimeMetric("load_model")
        start_time = time.time()
        self.encrypt_model(path_to_model)
        encrypt_model_metric.record(start_time, time.time())

        encrypt_data_metric = TimeMetric("encrypt_data")
        start_time = time.time()
        self.encrypt_data()
        encrypt_data_metric.record(start_time, time.time())

        evaluate_model_metric = TimeMetric("evaluate_model")
        start_time = time.time()
        self.evaluate()
        evaluate_model_metric.record(start_time, time.time())

        encrypt_model_metric.log()
        encrypt_data_metric.log()
        evaluate_model_metric.log()

    def encrypt_data(self):
        self.data_loader = PrivateMnistDataLoader(self.alice, self.bob, self.crypto_provider)

    def encrypt_model(self, path_to_model):
        self.model = torch.load(path_to_model)
        self.model.fix_precision().share(self.alice, self.bob, crypto_provider=self.crypto_provider)

    def evaluate(self):
        self.model.eval()
        private_correct_predictions = 0
        total_predictions = len(self.data_loader.private_test_loader) * TEST_BATCH_SIZE
        with torch.no_grad():
            for data, target in self.data_loader.private_test_loader:
                output = self.model(data)
                pred = output.argmax(dim=1)
                private_correct_predictions += pred.eq(target.view_as(pred)).sum()

            correct_predictions = private_correct_predictions.copy().get().float_precision().long().item()
            accuracy = 100.0 * correct_predictions / total_predictions
            print('Test set: Accuracy: {}/{} ({:.4f}%)'.format(correct_predictions, total_predictions, accuracy))
