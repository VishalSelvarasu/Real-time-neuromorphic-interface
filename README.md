# Real-Time Neuromorphic Gesture Recognition

> *CNNs vs SNNs — because the brain doesn't do matrix multiplications.*

SNNs are supposed to be the natural fit for event-camera data. I wanted to test whether that's actually true — or whether a conventional CNN still wins on a dataset that was literally designed for neuromorphic hardware.

The answer, at least at this stage, is complicated.

---

## What this project does

Compares a CNN baseline against an SNN on the DVSGesture dataset — IBM's event-camera gesture benchmark with 11 gesture classes recorded under different lighting conditions.

The question is simple: **does the architecture that was built for this type of data actually outperform one that wasn't?**

---

## Results

| Model | Test Accuracy | Avg Latency |
|-------|--------------|-------------|
| CNN   | 36.7%        | 20.85 ms    |
| SNN   | 23.9%        | 345.21 ms   |

The CNN wins on both metrics. That's not the result I expected going in.

**On accuracy:** Both numbers are low — DVSGesture has 11 classes, so random chance sits around 9%. The models are learning, but not confidently. The CNN has more mature tooling behind it; gradient flow through spiking neurons requires surrogate approximations that introduce their own noise, and getting that right is genuinely non-trivial.

**On latency:** 345ms for the SNN is the more surprising result. SNNs are supposed to be efficient — sparse, event-driven, low power. But that efficiency only shows up when the time step count and spike thresholds are properly tuned. The current setup runs more time steps than it probably needs to, which kills the latency advantage before it has a chance to appear.

**Honest take:** I don't fully understand every reason the SNN underperformed here. SNN training is an open research problem — surrogate gradient methods, threshold tuning, and temporal encoding are all areas where the field is still figuring things out. This project is my entry point into that, not a conclusion.

---

## Dataset

This project uses **DVSGesture** — an event-camera dataset from IBM Research, recorded with a DVS128 dynamic vision sensor.

The dataset is not included in the repository (too large for standard GitHub upload). Keep it locally at:

```
data/external/DVSGesture/
```

Expected structure:
```
ibmGestureTrain/
ibmGestureTest/
ibmGestureTrain.tar.gz
ibmGestureTest.tar.gz
```

---

## Repository structure

```
src/
  train_cnn.py         — CNN training script
  train_snn.py         — SNN training script
  evaluate.py          — Evaluation on test set
  latency_benchmark.py — Inference latency measurement
models/
  cnn_baseline.pt      — Saved CNN weights (included)
  snn_checkpoint.pt    — SNN weights (kept locally, too large for Git)
config.py              — Shared configuration
requirements.txt       — Python dependencies
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks script execution:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Running the project

```bash
# Train
python src\train_cnn.py
python src\train_snn.py

# Evaluate
python src\evaluate.py

# Latency benchmark
python src\latency_benchmark.py
```

---

## What's next

- **Time step tuning** — reducing SNN time steps should cut latency significantly without proportional accuracy loss
- **Spike threshold sweep** — systematic tuning of neuron thresholds is the most likely lever for accuracy improvement
- **Proper event-to-frame encoding** — the current preprocessing may not be extracting temporal structure optimally
- **Visualization** — spike raster plots to actually see what the SNN is responding to
- **Hardware target** — ultimately this is about robotics perception; testing on edge hardware (Jetson, or neuromorphic chips like Intel Loihi) is the logical next step

---

## Background

This sits at the intersection of two things I care about as a robotics student: real-time perception and energy-efficient inference. Event cameras are already showing up in robotics research for their low latency and high dynamic range. SNNs are the natural processing partner for that data — in theory.

This project is me stress-testing that theory with real data and being honest about what I found.

---

*DVSGesture dataset by IBM Research. Please check dataset terms before redistribution.*
