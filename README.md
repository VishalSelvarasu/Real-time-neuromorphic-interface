# Comparative Study of CNN and SNN Models for Real-Time Neuromorphic Gesture Recognition

This project compares a CNN baseline and an SNN model for real-time gesture recognition using the DVSGesture dataset.

I built it as a small research prototype to see how the two approaches behave in practice, especially around accuracy and latency.

## What’s in the repo

- `src/` has the training, evaluation, and benchmarking scripts.
- `models/` stores the saved model weights.
- `configs/` keeps project settings.
- `output/` is for plots, reports, and demo artifacts.
- `requirements.txt` lists the Python packages.

## Setup

Create a virtual environment and install the dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the script, use the execution policy command you already used earlier in the project.

## Run the project

Train the models:

```powershell
python src\train_cnn.py
python src\train_snn.py
```

Evaluate the models:

```powershell
python src\evaluate.py
```

Check latency:

```powershell
python src\latency_benchmark.py
```

## Dataset

This project uses **DVSGesture**.

The dataset is **not included in this repository** because it is too large for a normal GitHub upload. Keep it locally in:

```text
data/external/DVSGesture/
```

Expected files and folders:

- `ibmGestureTrain/`
- `ibmGestureTest/`
- `ibmGestureTrain.tar.gz`
- `ibmGestureTest.tar.gz`

If those folders are missing, the training and evaluation scripts won’t have anything to run on.

## Notes on the GitHub upload

The code is pushed to GitHub, but the dataset stays on your machine. That keeps the repository light and avoids GitHub file-size limits.

Git LFS is installed in this project, but the dataset itself is still better left out of the repo.

## Current results

| Model | Test Accuracy | Avg Latency |
|---|---:|---:|
| CNN | 36.7% | 20.85 ms |
| SNN | 23.9% | 345.21 ms |

The CNN is currently the stronger of the two in both accuracy and latency.

## Why this repo exists

I wanted a straightforward comparison between a conventional CNN and an SNN setup for neuromorphic gesture recognition.

It is not a polished benchmark suite; it is more of a practical project that shows the pipeline from dataset to training to evaluation.

## Project status

This is still a work in progress.

A few things that could improve it later:

- better tuning,
- cleaner plots,
- a nicer demo flow,
- and more careful SNN optimization.

## License and reuse

Please check the dataset rules separately before redistributing DVSGesture.

The code in this repo can be adapted for experiments, but the dataset itself should be downloaded from its original source.