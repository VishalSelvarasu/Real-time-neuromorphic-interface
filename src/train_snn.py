import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from config import DEVICE, MODEL_DIR, EPOCHS, LR
from utils import ensure_dir, seed_everything
from benchmark_loader import get_dvsgesture_dataloaders
from snn_model import EventSNN

seed_everything(42)
ensure_dir(MODEL_DIR)

checkpoint_path = MODEL_DIR / "snn_checkpoint.pt"

print("Loading DVSGesture dataloaders...")
train_loader, test_loader = get_dvsgesture_dataloaders()
print("Dataloaders ready.")

model = EventSNN(num_classes=11).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

start_epoch = 0

if checkpoint_path.exists():
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch + 1}")

print("Starting SNN training...")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE).float()
        y = y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss,
    }, checkpoint_path)

torch.save(model.state_dict(), MODEL_DIR / "snn_model.pt")
print("Saved SNN model")
