from typing import Dict


def invert_dict(in_dict):
    return {v: k for k, v in in_dict.items()}


def get_model_type(model_path: str, models_dict: Dict) -> str:
    model_types = invert_dict(models_dict)
    return model_types[model_path]
