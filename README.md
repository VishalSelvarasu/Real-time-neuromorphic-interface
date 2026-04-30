# Comparative Study of CNN and SNN Models for Real-Time Neuromorphic Gesture Recognition

This project compares a CNN baseline and an SNN model for real-time gesture recognition on the DVSGesture dataset.

The main goal is to observe how the two approaches differ in practice, especially in terms of accuracy and latency.

## Repository contents

- `src/` contains the training, evaluation, and latency benchmarking scripts.
- `models/` stores the saved model weights. The small CNN baseline is included in the repository, while the larger SNN checkpoint is kept locally.
- `requirements.txt` lists the Python dependencies.

## Setup

Create a virtual environment and install the dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks script execution, use the execution policy command already used earlier in the project.

## Running the project

Train the models:

```powershell
python src\train_cnn.py
python src\train_snn.py
```

Evaluate the models:

```powershell
python src\evaluate.py
```

Measure latency:

```powershell
python src\latency_benchmark.py
```

## Dataset

This project uses **DVSGesture**.

The dataset is not included in this project because it is too large for a standard GitHub upload. Keep it locally in:

```text
data/external/DVSGesture/
```

Expected files and folders:

- `ibmGestureTrain/`
- `ibmGestureTest/`
- `ibmGestureTrain.tar.gz`
- `ibmGestureTest.tar.gz`

If these folders are missing, the training and evaluation scripts will not have data to run on.

## GitHub contents

This project includes the code and lightweight assets only. The dataset archives and the full SNN checkpoint are kept out of Git history so the project remains smaller and easier to manage.

## Results

| Model | Test Accuracy | Avg Latency |
|---|---:|---:|
| CNN | 36.7% | 20.85 ms |
| SNN | 23.9% | 345.21 ms |

The current observation is that the CNN performs better than the SNN in both accuracy and latency under the present setup.

## Project overview

This project looks at how a conventional CNN and an SNN behave under the same gesture recognition workflow.

The observations from the current runs show a clear gap between the two models, particularly in inference speed and overall accuracy.

## Status

Completed baseline implementation.

Possible next steps include better tuning, cleaner plots, a more polished demo flow, and further SNN optimization.

## License and reuse

Please check the dataset terms separately before redistributing DVSGesture.

The code in this project can be adapted for experiments, but the dataset itself should be downloaded from its original source.
