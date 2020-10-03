import torch
import json


def torch2json(in_tensor):
    return json.dumps(in_tensor.tolist())


def json2torch(in_str):
    return torch.tensor(json.loads(in_str))


def json_data_loader(in_data_loader):
    """
    Encodes the test data as JSON.
    """
    return [(torch2json(data), labels) for data, labels in in_data_loader.test_loader]
