import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
CACHE_DIR = DATA_DIR / "cache"
DEMO_CACHE_DIR = DATA_DIR / "demo_cache"

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
DEMO_DIR = OUTPUT_DIR / "demo"

SENSOR_SIZE = (128, 128, 2)
NUM_CLASSES = 11
TIME_BINS = 8
BATCH_SIZE = 4
EPOCHS = 2
LR = 1e-3

FRAME_SIZE = (128, 128)
LIVE_EVENT_THRESHOLD = 18.0
LIVE_WINDOW = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"