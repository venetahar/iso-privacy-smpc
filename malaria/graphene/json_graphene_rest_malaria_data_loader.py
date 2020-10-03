from common.utils.pytorch_utils import json_data_loader
from malaria.common.malaria_data_loader import MalariaDataLoader


class JSONGrapheneRESTMalariaDataLoader(MalariaDataLoader):
    """
    A JSON Malaria Data loader responsible for encoding the data.
    """

    def __init__(self, data_path, test_batch_size):
        super(JSONGrapheneRESTMalariaDataLoader, self).__init__(data_path, test_batch_size)
        self.private_test_loader = []

    def encode_test_data(self):
        self.private_test_loader = json_data_loader(self.test_loader)
