import syft as sy
import torch

from common.private_inference import PrivateInference


class PysyftPrivateInference(PrivateInference):
    """
    Class encapsulating the logic for performing private inference using PySyft.
    """

    def __init__(self, data_loader, parameters=None):
        """
        Returns a PysyftPrivateInference object.
        :param data_loader: The data loader.
        :param parameters: Any additional parameters for inference.
        """
        hook = sy.TorchHook(torch)
        self.client = sy.VirtualWorker(hook, id="client")
        self.bob = sy.VirtualWorker(hook, id="bob")
        self.alice = sy.VirtualWorker(hook, id="alice")
        self.crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
        self.data_loader = data_loader
        self.parameters = parameters
        self.model = None

    def encrypt_data(self):
        """
        Encrypts the data.
        """
        self.data_loader.encrypt_test_data(self.alice, self.bob, self.crypto_provider)

    def encrypt_model(self, path_to_model):
        """
        Encrypts the model.
        :param path_to_model: The path to the saved model to be loaded and secret shared.
        """
        self.model = torch.load(path_to_model)
        self.model.fix_precision().share(self.alice, self.bob, crypto_provider=self.crypto_provider)

    def evaluate(self):
        """
        Performs secure evaluation of the model.
        """
        self.model.eval()
        private_correct_predictions = 0
        total_predictions = len(self.data_loader.private_test_loader) * self.parameters['test_batch_size']
        with torch.no_grad():
            for batch_index, (data, target) in enumerate(self.data_loader.private_test_loader):
                print("Performing inference for batch {}".format(batch_index))
                output = self.model(data)
                pred = output.argmax(dim=1)
                private_correct_predictions += pred.eq(target.view_as(pred)).sum()

            correct_predictions = private_correct_predictions.copy().get().float_precision().long().item()
            accuracy = 100.0 * correct_predictions / total_predictions
            print('Test set: Accuracy: {}/{} ({:.4f}%)'.format(correct_predictions, total_predictions, accuracy))
