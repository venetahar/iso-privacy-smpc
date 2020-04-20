import time


class TimeMetric:

    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.value = None
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            print("start method needs to be called first.")
        else:
            self.value = time.time() - self.start_time

    def log(self):
        if self.value is not None:
            print("Metric {} took {} time to execute".format(self.metric_name, self.value))
        else:
            print("Metric {} has no value".format(self.metric_name))
