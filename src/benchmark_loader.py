import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tonic
from tonic import transforms
from torch.utils.data import DataLoader
from config import EXTERNAL_DIR, CACHE_DIR, TIME_BINS, BATCH_SIZE
from utils import ensure_dir

def get_dvsgesture_dataloaders():
    ensure_dir(EXTERNAL_DIR)
    ensure_dir(CACHE_DIR)

    sensor_size = tonic.datasets.DVSGesture.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=TIME_BINS)
    ])

    train_set = tonic.datasets.DVSGesture(
        save_to=str(EXTERNAL_DIR),
        train=True,
        transform=frame_transform
    )

    test_set = tonic.datasets.DVSGesture(
        save_to=str(EXTERNAL_DIR),
        train=False,
        transform=frame_transform
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dvsgesture_dataloaders()
    x, y = next(iter(train_loader))
    print("Train batch shape:", x.shape)
    print("Labels shape:", y.shape)