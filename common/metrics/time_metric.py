import time


class TimeMetric:
    """
    A time metric.
    """

    def __init__(self, metric_name):
        """
        Creates a time metric object.
        :param metric_name: The name of the metric.
        """
        self.metric_name = metric_name
        self.value = None
        self.start_time = None

    def start(self):
        """
        Starts recording the time.
        """
        self.start_time = time.time()

    def stop(self):
        """
        Stops recording if previously started.
        :return:
        """
        if self.start_time is None:
            print("start method needs to be called first.")
        else:
            self.value = time.time() - self.start_time

    def log(self):
        """
        Logs the metric if there is a valid value.
        """
        if self.value is not None:
            print("Metric {} took {} time to execute".format(self.metric_name, self.value))
        else:
            print("Metric {} has no value".format(self.metric_name))
