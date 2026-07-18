# MLOps Pipeline – Facial Emotion Recognition

A full MLOps pipeline for training, evaluating, and deploying a facial emotion 
recognition model. The focus of this project is the pipeline infrastructure, 
not model performance — the CNN architecture was developed separately in 
[deeplearningmini](https://github.com/carlemil112/deeplearningmini).

## Pipeline overview
```
Git push → Jenkins → Docker build → Unit tests → DVC pull (MinIO)
→ DeepSpeed training → Post-training quantization (ONNX)
→ Model evaluation → Auto-merge to main → MLflow logging
```

## What the pipeline does

- **Jenkins CI/CD** — parameterised pipeline with toggleable stages (training, 
  evaluation, model registration)
- **Docker** — builds and tags image per commit SHA, pushes to private registry
- **DVC + MinIO** — versioned dataset storage and retrieval
- **DeepSpeed** — distributed training with ZeRO optimization on GPU
- **Post-training quantization** — INT8 ONNX conversion for inference efficiency
- **MLflow** — experiment tracking and model registry
- **Auto-merge** — successful builds on `development` branch auto-merge to `main`

## Model

3-block CNN trained on FER2013 (facial emotion recognition, 5 classes):
- Block 1: Conv2d(1→64) + BatchNorm + ReLU + MaxPool + Dropout
- Block 2: Conv2d(64→128) + BatchNorm + ReLU + MaxPool + Dropout  
- Block 3: Conv2d(128→128) + BatchNorm + ReLU + MaxPool + Dropout
- Classifier: Flatten → Linear(4608→256) → Linear(256→5)

**Original model performance** (full dataset, deeplearningmini):
- Training accuracy: 75.2%
- Validation accuracy: 64.3%

**Pipeline model performance** (reduced dataset for pipeline testing):
- Training accuracy: 17.3%
- Validation accuracy: 15.1%

> Note: The reduced accuracy reflects intentional dataset reduction for pipeline 
> testing purposes. The pipeline infrastructure is the deliverable, not model 
> performance.

## Tech stack
- **PyTorch** — model definition and training
- **DeepSpeed** — ZeRO distributed training optimization
- **Jenkins** — CI/CD orchestration
- **Docker** — containerised and reproducible builds
- **DVC + MinIO** — data versioning and remote storage
- **MLflow** — experiment tracking and model registry
- **ONNX** — model export and INT8 quantization
