import syft as sy
import torch

from common.private_inference import PrivateInference


class PysyftPrivateInference(PrivateInference):
    """
    Class encapsulating the logic for performing private inference using PySyft.
    """

    def __init__(self, test_data_loader, parameters=None):
        """
        Returns a PysyftPrivateInference object.
        :param test_data_loader: The data loader.
        :param parameters: Any additional parameters for inference.
        """
        super(PysyftPrivateInference, self).__init__(test_data_loader, parameters)
        hook = sy.TorchHook(torch)
        self.bob = sy.VirtualWorker(hook, id="bob")
        self.alice = sy.VirtualWorker(hook, id="alice")
        self.crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    def encrypt_data(self):
        """
        Encrypts the data.
        """
        self.test_data_loader.encrypt_test_data(self.alice, self.bob, self.crypto_provider)

    def encrypt_single_instance(self, data_instance):
        """
        Returns an encrypted version of the data instance.
        :param data_instance: The data instance.
        :return: An encrypted version of the data.
        """
        return data_instance.fix_prec().share(self.alice, self.bob, crypto_provider=self.crypto_provider)

    def encrypt_model(self, path_to_model):
        """
        Encrypts the model.
        :param path_to_model: The path to the saved model to be loaded and secret shared.
        """
        self.private_model = torch.load(path_to_model)
        self.private_model.fix_precision().share(self.alice, self.bob, crypto_provider=self.crypto_provider)

    def evaluate(self):
        """
        Performs secure evaluation of the model.
        """
        self.private_model.eval()
        private_correct_predictions = 0
        total_predictions = len(self.test_data_loader.private_test_loader) * self.parameters['test_batch_size']
        with torch.no_grad():
            for batch_index, (data, target) in enumerate(self.test_data_loader.private_test_loader):
                print("Performing inference for batch {}".format(batch_index))
                output = self.private_model(data)
                pred = output.argmax(dim=1)
                private_correct_predictions += pred.eq(target.view_as(pred)).sum()

            correct_predictions = private_correct_predictions.copy().get().float_precision().long().item()
            accuracy = 100.0 * correct_predictions / total_predictions
            print('Test set: Accuracy: {}/{} ({:.4f}%)'.format(correct_predictions, total_predictions, accuracy))

    def measure_communication_costs(self, path_to_model):
        """
        Measures the communication costs of performing private inference.
        :param path_to_model: The path to the trained model.
        """
        self.alice.log_msgs = True
        self.bob.log_msgs = True
        self.crypto_provider.log_msgs = True

        # Just in case reset the message history.
        self.alice.msg_history = list()
        self.bob.msg_history = list()
        self.crypto_provider.msg_history = list()

        super().measure_communication_costs(path_to_model)

        alice_total_bytes = self._count_bytes(self.alice)
        bob_total_bytes = self._count_bytes(self.bob)
        crypto_total_bytes = self._count_bytes(self.crypto_provider)

        print("=====Communication costs: =====")
        print("Alice total bytes: {}".format(alice_total_bytes))
        print("Bob total bytes: {}".format(bob_total_bytes))
        print("Crypto provider total bytes: {}".format(crypto_total_bytes))

    @staticmethod
    def _count_bytes(worker):
        """
        Counts the number of bytes. As messages in PySyft seem to be bytes objects we can use the length to determine
        the number of bytes per message:
        https://en.wikiversity.org/wiki/Python_Concepts/Bytes_objects_and_Bytearrays#bytes_objects
        :param worker: The worker.
        :return: The total bytes for this worker.
        """
        total_bytes = 0
        for msg in worker.msg_history:
            total_bytes += len(msg)
        return total_bytes
