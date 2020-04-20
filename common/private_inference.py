from common.metrics.time_metric import TimeMetric
import numpy as np


class PrivateInference:
    """
    A base class holding common functionality for private inference.
    """

    def __init__(self, test_data_loader, parameters):
        self.test_data_loader = test_data_loader
        self.parameters = parameters
        self.private_model = None

    def perform_inference(self, path_to_model):
        """
        Performs private inference and prints the final accuracy.
        :param path_to_model: The path to the saved model.
        """
        self.encrypt_model(path_to_model)
        self.encrypt_data()
        self.evaluate()

    def measure_runtime(self, path_to_model, num_runs=20):
        all_runs_metrics = {
            'encrypt_model_metrics': [],
            'encrypt_data_metrics': [],
            'evaluate_model_metric': []
        }
        data, _ = next(iter(self.test_data_loader.test_loader))
        for index in range(0, num_runs):
            encrypt_model_metric = TimeMetric("encrypt_model")
            encrypt_model_metric.start()
            self.encrypt_model(path_to_model)
            encrypt_model_metric.stop()
            all_runs_metrics['encrypt_model_metrics'].append(encrypt_model_metric.value)

            encrypt_data_metric = TimeMetric("encrypt_data")
            encrypt_data_metric.start()
            encrypted_data = self.encrypt_single_instance(data)
            encrypt_data_metric.stop()
            all_runs_metrics['encrypt_data_metrics'].append(encrypt_data_metric.value)

            evaluate_model_metric = TimeMetric("evaluate_model")
            evaluate_model_metric.start()
            self.private_model(encrypted_data)
            evaluate_model_metric.stop()
            all_runs_metrics['evaluate_model_metric'].append(evaluate_model_metric.value)
        print("============Performance metrics: ============ ")
        print("Average encrypt model time: {}".format(np.mean(all_runs_metrics['encrypt_model_metrics'])))
        print("Average encrypt data time: {}".format(np.mean(all_runs_metrics['encrypt_data_metrics'])))
        print("Average evaluate model time: {}".format(np.mean(all_runs_metrics['evaluate_model_metric'])))

    def measure_communication_costs(self, path_to_model):
        data, _ = next(iter(self.test_data_loader.test_loader))
        self.encrypt_model(path_to_model)
        encrypted_data = self.encrypt_single_instance(data)
        self.private_model(encrypted_data)

    def encrypt_model(self, path_to_model):
        print("Not implemented")
        pass

    def encrypt_data(self):
        print("Not implemented")
        pass

    def encrypt_single_instance(self, data_instance):
        print("Not implemented")
        pass

    def evaluate(self):
        print("Not implemented")
        pass
