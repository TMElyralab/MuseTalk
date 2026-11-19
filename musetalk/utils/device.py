import os

import torch


def get_device(gpu_id=0):
    """Return the torch device to run on.

    Order: the MUSETALK_DEVICE environment variable if set (for example
    "cpu", "xpu", "mps" or "cuda:1"), then CUDA, then Intel XPU, then CPU.
    """
    override = os.environ.get("MUSETALK_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device(f"xpu:{gpu_id}")
    return torch.device("cpu")
