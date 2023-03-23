import torch

def get_torch_dtype(dtype_name):
    return getattr(torch, dtype_name)