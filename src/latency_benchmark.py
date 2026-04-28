import sys
from pathlib import Path
import time
import torch

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from config import DEVICE as DEVICE_STR, MODEL_DIR
from benchmark_loader import get_dvsgesture_dataloaders
from frame_baseline import FrameBaselineCNN
from snn_model import EventSNN

DEVICE = torch.device(DEVICE_STR) if isinstance(DEVICE_STR, str) else DEVICE_STR

print("Loading test dataloader...")
_, test_loader = get_dvsgesture_dataloaders()

print("Loading trained models...")
cnn = FrameBaselineCNN(num_classes=11).to(DEVICE)
cnn.load_state_dict(torch.load(MODEL_DIR / "cnn_baseline.pt", map_location=DEVICE))
cnn.eval()

snn = EventSNN(num_classes=11).to(DEVICE)
snn.load_state_dict(torch.load(MODEL_DIR / "snn_model.pt", map_location=DEVICE))
snn.eval()

cnn_times = []
snn_times = []

print("Benchmarking inference latency on up to 100 test batches...")

with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        if i >= 100:
            break

        x = x.to(DEVICE).float()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = cnn(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        cnn_times.append(time.perf_counter() - start)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = snn(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        snn_times.append(time.perf_counter() - start)

cnn_avg = sum(cnn_times) / len(cnn_times)
snn_avg = sum(snn_times) / len(snn_times)

print(f"CNN Avg Latency: {cnn_avg * 1000:.2f} ms")
print(f"SNN Avg Latency: {snn_avg * 1000:.2f} ms")
print(f"SNN/CNN Latency Ratio: {snn_avg / cnn_avg:.2f}x slower")
print(f"CNN/SNN Speed Ratio: {snn_avg / cnn_avg:.2f}x faster")