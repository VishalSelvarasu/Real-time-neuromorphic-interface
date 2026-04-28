from pathlib import Path
import time
import random
import numpy as np
import torch

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def elapsed_ms(self):
        return (time.perf_counter() - self.start_time) * 1000.0