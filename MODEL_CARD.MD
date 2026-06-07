---
language:
  - en
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - facial-emotion-recognition
  - pytorch
  - deepspeed
datasets:
  - fer2013
metrics:
  - accuracy
co2_eq_emissions:
  emissions: TBA
  source: CarbonTracker
  training_type: pre-training
  geographic_location: Denmark
  hardware_used: NVIDIA GPU
model-index:
  - name: FER Emotion Recognition CNN
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: FER2013
          type: fer2013
        metrics:
          - type: accuracy
            name: Validation Accuracy
            value: 0.151
---

# Model Card: FER Emotion Recognition CNN

## Model Description

This model is a convolutional neural network (CNN) trained to classify facial expressions into seven emotion categories. It was developed as part of an MLOps course project at Aalborg University.

- **Model type:** Convolutional Neural Network (CNN)
- **Task:** Facial Emotion Recognition (multi-class classification)
- **Input:** Grayscale facial images, 48×48 pixels
- **Output:** Probability distribution over 7 emotion classes
- **Framework:** PyTorch


## Training Data

The model was trained on the FER2013 dataset, a publicly available benchmark dataset for facial emotion recognition. The dataset consists of 48×48 pixel grayscale images of faces, labeled with one of seven emotions. Data augmentation and class balancing were applied during preprocessing to address class imbalance.

- **Dataset:** FER2013
- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Preprocessing:** Resize to 48×48, normalization (mean=0.5147, std=0.2536)
- **Split:** 80% training, 20% validation

## Architecture

Three convolutional blocks followed by a fully connected classifier:

| Layer | Details |
|---|---|
| Block 1 | Conv2d(1→64), BN, ReLU, Conv2d(64→64), BN, ReLU, MaxPool2d, Dropout(0.4) |
| Block 2 | Conv2d(64→128), BN, ReLU, Conv2d(128→128), BN, ReLU, MaxPool2d, Dropout(0.4) |
| Block 3 | Conv2d(128→128), BN, ReLU, MaxPool2d, Dropout(0.4) |
| Classifier | Flatten, Linear(4608→256), BN, ReLU, Dropout(0.5), Linear(256→7) |

Total parameters: ~6M

## Performance

| Metric | Value |
|---|---|
| Validation Accuracy | 15.1% |
| Training Accuracy | 17.3% |


## MLOps Pipeline

The model is trained and versioned through an automated Jenkins CI/CD pipeline:

- Data versioning via DVC with MinIO S3 storage
- Distributed training with DeepSpeed ZeRO 
- Experiment tracking and model registry via MLflow
- Inference with a subset of the test split

## Limitations and Biases

- FER2013 is known to contain labeling noise, which may affect model reliability
- Performance may vary across demographic groups due to dataset composition
- The model is trained on static images and may not generalize to video frames or non-frontal faces
- Emotion recognition is inherently subjective and culturally dependent

## Ethical Considerations

Facial emotion recognition systems carry significant ethical risks, including potential misuse for surveillance or discrimination. This model is intended strictly for academic purposes and should not be deployed in contexts where automated emotion inference could harm individuals.
