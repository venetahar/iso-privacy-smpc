import time

import syft as sy
import torch

from common.metrics.time_metric import TimeMetric


class PysyftPrivateInference:

    def __init__(self, data_loader, parameters=None):
        hook = sy.TorchHook(torch)
        self.client = sy.VirtualWorker(hook, id="client")
        self.bob = sy.VirtualWorker(hook, id="bob")
        self.alice = sy.VirtualWorker(hook, id="alice")
        self.crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
        self.data_loader = data_loader
        self.parameters = parameters
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
        self.data_loader.encrypt_data(self.alice, self.bob, self.crypto_provider)

    def encrypt_model(self, path_to_model):
        self.model = torch.load(path_to_model)
        self.model.fix_precision().share(self.alice, self.bob, crypto_provider=self.crypto_provider)

    def evaluate(self):
        self.model.eval()
        private_correct_predictions = 0
        total_predictions = len(self.data_loader.private_test_loader) * self.parameters['test_batch_size']
        with torch.no_grad():
            for data, target in self.data_loader.private_test_loader:
                print(data)
                output = self.model(data)
                pred = output.argmax(dim=1)
                private_correct_predictions += pred.eq(target.view_as(pred)).sum()

            correct_predictions = private_correct_predictions.copy().get().float_precision().long().item()
            accuracy = 100.0 * correct_predictions / total_predictions
            print('Test set: Accuracy: {}/{} ({:.4f}%)'.format(correct_predictions, total_predictions, accuracy))
