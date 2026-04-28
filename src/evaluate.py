import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.metrics import classification_report, accuracy_score
from config import DEVICE, MODEL_DIR
from benchmark_loader import get_dvsgesture_dataloaders
from frame_baseline import FrameBaselineCNN
from snn_model import EventSNN

def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE).float()
            out = model(x)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    return accuracy_score(y_true, y_pred), classification_report(y_true, y_pred)

if __name__ == "__main__":
    _, test_loader = get_dvsgesture_dataloaders()

    cnn = FrameBaselineCNN(num_classes=11).to(DEVICE)
    cnn.load_state_dict(torch.load(MODEL_DIR / "cnn_baseline.pt", map_location=DEVICE))

    snn = EventSNN(num_classes=11).to(DEVICE)
    snn.load_state_dict(torch.load(MODEL_DIR / "snn_model.pt", map_location=DEVICE))

    cnn_acc, cnn_report = evaluate_model(cnn, test_loader)
    snn_acc, snn_report = evaluate_model(snn, test_loader)

    print("\nCNN Accuracy:", cnn_acc)
    print(cnn_report)

    print("\nSNN Accuracy:", snn_acc)
    print(snn_report)