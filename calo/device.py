"""Explicit compute-device selection shared by training and evaluation."""

import torch


def select_device(requested="auto"):
    requested = str(requested).lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device: {requested}")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available to this Python process")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
