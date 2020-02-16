import syft as sy
import torch

from common.constants import TEST_BATCH_SIZE
from pysyft.private_mnist_data_loader import PrivateMnistDataLoader


class SMPCMnist:

    def __init__(self):
        hook = sy.TorchHook(torch)
        self.client = sy.VirtualWorker(hook, id="client")
        self.bob = sy.VirtualWorker(hook, id="bob")
        self.alice = sy.VirtualWorker(hook, id="alice")
        self.crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
        self.data_loader = PrivateMnistDataLoader(self.alice, self.bob, self.crypto_provider)
        self.model = None

    def encrypt_evaluate_model(self, path_to_model):
        self.load_model(path_to_model)
        self.model.fix_precision().share(self.alice, self.bob, crypto_provider=self.crypto_provider)
        self.evaluate(self.model)

    def load_model(self, path_to_model):
        self.model = torch.load(path_to_model)

    def evaluate(self, encrypted_model):
        encrypted_model.eval()
        private_correct_predictions = 0
        total_predictions = len(self.data_loader.private_test_loader) * TEST_BATCH_SIZE
        with torch.no_grad():
            for data, target in self.data_loader.private_test_loader:
                output = encrypted_model(data)
                pred = output.argmax(dim=1)
                private_correct_predictions += pred.eq(target.view_as(pred)).sum()

            correct_predictions = private_correct_predictions.copy().get().float_precision().long().item()
            accuracy = 100.0 * correct_predictions / total_predictions
            print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct_predictions, total_predictions, accuracy))
