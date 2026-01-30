# Image Classification with ResNet18 Transfer Learning to Dogs and Cats

This project implements **binary image classification** using a **pretrained ResNet18 model** from PyTorch. The model is fine-tuned on your dataset, training only the last layers while freezing earlier layers.

---

## Features

- Uses **ResNet18** pretrained on ImageNet.
- **Transfer learning** with frozen initial layers for faster training.
- Fine-tunes **layer3, layer4, and fully connected (fc) layers**.
- **Custom learning rates** for different layers.
- Implements **StepLR scheduler** for learning rate decay.
- Evaluates test accuracy after training.


## Usage

Update the TRAIN_DIR and TEST_DIR paths in the code to point to your dataset directories.

Run the script:

python train_resnet18.py

## The script will:

- Load training and test datasets.
- Apply data augmentation (RandomResizedCrop, RandomHorizontalFlip, normalization).
- Train the model for 10 epochs.
- Print training loss per epoch.
- Evaluate test accuracy on the test dataset.

## Training Details

- Device: Automatically uses GPU if available.
- Optimizer: Adam with separate learning rates:
  - layer3 and layer4 → 1e-4
  - Fully connected layer → 1e-3
- Loss function: CrossEntropyLoss
- Scheduler: StepLR with step_size=10, gamma=0.1
- Batch size: 32
- Image size: 224x224

## Scope for Improvement

- Use better data augmentation for better generalization.
- Train for more epochs or unfreeze additional layers.
- Experiment with other pretrained models like ResNet50, EfficientNet, etc.
- Add early stopping or model checkpointing.

## Author

Tejas Arjuna Ashok Kumar
