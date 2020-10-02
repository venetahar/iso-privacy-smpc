from common.private_inference import PrivateInference
import requests

from common.utils.pytorch_utils import *


class GrapheneRESTClientPrivateInference(PrivateInference):

    def __init__(self, service_url, test_data_loader, parameters=None):
        """
        Creates a GrapheneRESTPrivateInference.
        :param test_data_loader: The data_loader.
        :param parameters: The parameters.
        """
        super(GrapheneRESTClientPrivateInference, self).__init__(test_data_loader, parameters)
        self.service_url = f"{service_url}/api/predict_batch"
        self.model = "mnist_fc_model"
        # hack to accommodate benchmarking framework's API :)
        self.private_model = self.predict

    def predict(self, input_data):
        data_field = {
            "batch": input_data,
            "model": self.model
        }
        output_response = requests.post(self.service_url, data=data_field)

        pred = json2torch(output_response.text)
        return pred

    def encrypt_data(self):
        """
        Encodes the data.
        """
        self.test_data_loader.encode_test_data()

    def encrypt_single_instance(self, data_instance):
        return torch2json(data_instance)

    def encrypt_model(self, path_to_model):
        self.model = path_to_model

    def evaluate(self):
        import torch

        private_correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch_index, (data, target) in enumerate(self.test_data_loader.private_test_loader):
                print("Performing inference for batch {}".format(batch_index))

                pred = self.predict(data)
                # done upstream on server to avoid disclosing prediction values
                # pred = output.argmax(dim=1)

                private_correct_predictions += pred.eq(target.view_as(pred)).sum()
                total_predictions += len(target)

            correct_predictions = private_correct_predictions.item()
            accuracy = 100.0 * correct_predictions / total_predictions
            print('Test set: Accuracy: {}/{} ({:.4f}%)'.format(correct_predictions, total_predictions, accuracy))
