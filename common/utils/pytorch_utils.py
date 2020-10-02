import torch
import json


def torch2json(in_tensor):
    return json.dumps(in_tensor.tolist())


def json2torch(in_str):
    return torch.tensor(json.loads(in_str))

