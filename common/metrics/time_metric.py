class TimeMetric:

    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.value = None

    def record(self, start_time, end_time):
        self.value = end_time - start_time

    def log(self):
        if self.value is not None:
            print("Metric {} took {} time to execute".format(self.metric_name, self.value))
        else:
            print("Metric {} has no value".format(self.metric_name))
