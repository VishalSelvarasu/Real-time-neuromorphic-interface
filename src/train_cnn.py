import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from config import DEVICE, MODEL_DIR, EPOCHS, LR
from utils import ensure_dir, seed_everything
from benchmark_loader import get_dvsgesture_dataloaders
from frame_baseline import FrameBaselineCNN

seed_everything(42)
ensure_dir(MODEL_DIR)

print("Loading DVSGesture dataloaders...")
train_loader, test_loader = get_dvsgesture_dataloaders()
print("Dataloaders ready.")

model = FrameBaselineCNN(num_classes=11).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("Starting CNN training...")

for epoch in range(EPOCHS):
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

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")

torch.save(model.state_dict(), MODEL_DIR / "cnn_baseline.pt")
print("Saved CNN baseline model")