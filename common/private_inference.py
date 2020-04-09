import time

from common.metrics.time_metric import TimeMetric


class PrivateInference:
    """
    A base class holding common functionality for private inference.
    """

    def perform_inference(self, path_to_model):
        """
        Performs private inference and prints the final accuracy.
        :param path_to_model: The path to the saved model.
        """
        encrypt_model_metric = TimeMetric("encrypt_model")
        start_time = time.time()
        self.encrypt_model(path_to_model)
        encrypt_model_metric.record(start_time, time.time())
        encrypt_model_metric.log()

        encrypt_data_metric = TimeMetric("encrypt_data")
        start_time = time.time()
        self.encrypt_data()
        encrypt_data_metric.record(start_time, time.time())
        encrypt_data_metric.log()

        evaluate_model_metric = TimeMetric("evaluate_model")
        start_time = time.time()
        self.evaluate()
        evaluate_model_metric.record(start_time, time.time())
        evaluate_model_metric.log()

    def encrypt_model(self, path_to_model):
        print("Not implemented")
        pass

    def encrypt_data(self):
        print("Not implemented")
        pass

    def evaluate(self):
        print("Not implemented")
        pass