# 😷 Face Mask Detection Using CNN & PyTorch

## Deep Learning Image Classification Project

This project implements a **Face Mask Detection system using Deep Learning and Convolutional Neural Networks (CNNs)** with **PyTorch**.

The objective of the project is to automatically classify facial images into two categories:

* 😷 **With Mask**
* 🙂 **Without Mask**

A custom CNN architecture is developed from scratch and trained on the **Face Mask 12K Images Dataset**. 
The project covers the complete deep learning workflow, including dataset loading, image preprocessing, data augmentation, CNN architecture design, GPU/CPU handling, model training, validation and prediction.

---

## 📌 Project Overview

Face mask detection is a computer vision classification problem in which a model analyzes an image of a person's face and determines whether the person is wearing a mask.

Instead of relying on manually engineered image features, this project uses a **Convolutional Neural Network (CNN)** to automatically learn useful visual features directly from training images.

The overall workflow is:

```text
Face Mask Image Dataset
        ↓
Load Images
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
PyTorch DataLoader
        ↓
Custom CNN Architecture
        ↓
Model Training
        ↓
Validation
        ↓
Prediction
        ↓
With Mask / Without Mask
```

---

# 🎯 Project Objective

The primary objective of this project is to build an image classification model capable of distinguishing between faces **with masks** and faces **without masks**.

The project demonstrates practical knowledge of:

* Computer Vision
* Deep Learning
* Convolutional Neural Networks
* Image preprocessing
* Image augmentation
* PyTorch
* Model training and validation
* GPU-based deep learning
* Classification accuracy evaluation
* Prediction on unseen images

---

# 📂 Dataset

The project uses the **Face Mask 12K Images Dataset**.

### Dataset Source

Kaggle:

`Face Mask 12K Images Dataset`

The dataset is organized into separate directories for:

```text
Face Mask Dataset/
│
├── Train/
│   ├── WithMask/
│   └── WithoutMask/
│
├── Validation/
│   ├── WithMask/
│   └── WithoutMask/
│
└── Test/
    ├── WithMask/
    └── WithoutMask/
```

The directory structure makes it convenient to use PyTorch's `ImageFolder` utility, which automatically assigns labels based on folder names.

---

# 🧠 Technologies Used

| Technology             | Purpose                                        |
| ---------------------- | ---------------------------------------------- |
| Python                 | Main programming language                      |
| PyTorch                | Deep learning framework                        |
| Torchvision            | Image datasets, transformations and utilities |
| CNN                    | Image classification model                     |
| PIL                    | Image loading and inspection                   |
| Matplotlib             | Image visualization                            |
| NumPy/Python utilities | Supporting data operations                     |
| CUDA                   | GPU acceleration when available                |
| Jupyter Notebook       | Model development and experimentation          |

---

# 📚 Python Libraries

The project primarily uses:

```python
import torch
import torchvision
import matplotlib.pyplot as plt
import os

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from time import time

import torchvision.transforms as tt
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
```

---

# 🔍 Step 1 — Import Required Libraries

The first step is importing all libraries required for:

* Tensor operations
* Deep learning
* CNN construction
* Dataset management
* Image transformations
* Batch processing
* Image visualization

PyTorch serves as the primary deep learning framework throughout the project.

---

# 📁 Step 2 — Define Dataset Paths

Separate paths are defined for the training, testing and validation datasets.

```python
train_path = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Train'

test_path = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Test'

val_path = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Validation'
```

The project also checks whether these directories exist before proceeding.

This helps verify that the dataset has been loaded correctly into the working environment.

---

# 🖼️ Step 3 — Inspect the Images

An individual image from the dataset is loaded using PIL.

```python
image = Image.open(
    '../input/face-mask-12k-images-dataset/Face Mask Dataset/Train/WithMask/10.png'
)

print(image.size)
```

This step allows us to inspect the original image dimensions before preprocessing.

Since neural networks require images to have consistent dimensions, resizing is performed during preprocessing.

---

# 🔄 Step 4 — Image Preprocessing & Data Augmentation

One of the most important stages of a computer vision project is preparing the images before feeding them into the neural network.

The training transformations used in this project are:

```python
train_tfms = tt.Compose([
    tt.Resize([128, 128]),
    tt.RandomHorizontalFlip(),
    tt.ColorJitter(),
    tt.ToTensor(),
    tt.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
])
```

### Resize

```python
tt.Resize([128, 128])
```

Every image is resized to:

```
128 × 128 pixels
```

This ensures that every input to the CNN has the same dimensions.

---

### Random Horizontal Flip

```python
tt.RandomHorizontalFlip()
```

Training images are randomly flipped horizontally.

This acts as **data augmentation**, helping the model learn features that are less dependent on the orientation of the original image.

---

### Color Jitter

```python
tt.ColorJitter()
```

Color properties of training images are randomly modified.

This helps the model become more robust to variations in:

* Lighting
* Color
* Contrast
* Image conditions

---

### Convert Image to Tensor

```python
tt.ToTensor()
```

Images are converted into PyTorch tensors so they can be processed by the neural network.

---

### Image Normalization

```python
tt.Normalize(
    [0.4914, 0.4822, 0.4465],
    [0.2023, 0.1994, 0.2010]
)
```

Normalization transforms the input pixel distributions to improve numerical stability during neural-network training.

---

# 🧪 Validation & Test Transformations

Unlike the training dataset, random augmentation is not applied to the validation and test datasets.

The validation and testing pipeline contains:

```
Resize
   ↓
Convert to Tensor
   ↓
Normalize
```

This allows model performance to be evaluated consistently.

---

# 📦 Step 5 — Load Dataset Using ImageFolder

PyTorch's `ImageFolder` class is used to load images from their respective directories.

```python
train_ds = ImageFolder(
    '../input/face-mask-12k-images-dataset/Face Mask Dataset/Train',
    train_tfms
)

val_ds = ImageFolder(
    '../input/face-mask-12k-images-dataset/Face Mask Dataset/Validation',
    val_tfms
)

test_ds = ImageFolder(
    '../input/face-mask-12k-images-dataset/Face Mask Dataset/Test',
    test_tfms
)
```

`ImageFolder` automatically identifies the classes based on directory names.

Therefore, the model learns to distinguish between:

```
WithMask
WithoutMask
```

---

# 📦 Step 6 — Create DataLoaders

Loading all images simultaneously would consume a large amount of memory.

Instead, PyTorch `DataLoader` is used to process images in batches.

The training batch size is:

```python
batch_size = 64
```

Training DataLoader:

```python
train_dl = DataLoader(
    train_ds,
    batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

Validation DataLoader:

```python
val_dl = DataLoader(
    val_ds,
    batch_size * 2,
    num_workers=4,
    pin_memory=True
)
```

Testing is performed with:

```python
test_dl = DataLoader(
    test_ds,
    batch_size=1
)
```

### Why Shuffle Training Data?

```python
shuffle=True
```

Randomizing the training examples prevents the model from learning patterns associated with the order of the dataset.

---

# 👀 Step 7 — Visualize Training Images

Before training, batches of images are visualized using:

```python
make_grid()
```

and:

```python
matplotlib
```

A grid containing up to 64 images is displayed.

This is an important sanity check that helps verify:

* Images were loaded successfully
* Transformations were applied
* Tensor dimensions are correct
* Dataset structure is valid

---

# ⚡ Step 8 — GPU / CPU Configuration

Deep learning training can be computationally expensive.

The notebook checks whether CUDA is available.

```python
if torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
```

The project also defines a reusable function:

```python
def to_device(data, device):
```

to transfer data to the selected processing device.

A custom DataLoader wrapper is implemented to automatically move batches to the selected device during iteration.

Conceptually:

```text
Dataset
   ↓
DataLoader
   ↓
Device DataLoader
   ↓
GPU / CPU
   ↓
CNN
```

This design makes the training pipeline easier to run on different hardware environments.

---

# 🧠 Step 9 — Accuracy Calculation

A custom accuracy function is created:

```python
def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)

    return torch.tensor(
        torch.sum(preds == labels).item() / len(preds)
    )
```

The function:

1. Finds the class with the highest model output.
2. Compares predictions against actual labels.
3. Calculates the percentage of correct predictions.

---

# 🏗️ Step 10 — Base Model Class

A reusable base model class named:

```python
FaceMaskDetec
```

is implemented.

It contains functions for:

### Training Step

```python
training_step()
```

Calculates the training loss.

### Validation Step

```python
validation_step()
```

Calculates:

* Validation loss
* Validation accuracy

### Validation Epoch End

```python
validation_epoch_end()
```

Combines validation metrics from multiple batches.

### Epoch End

```python
epoch_end()
```

Displays training statistics after each epoch.

Example format:

```text
Epoch [0],
train_loss: ...
val_loss: ...
val_acc: ...
```

This modular structure separates training logic from the CNN architecture.

---

# 🧠 Step 11 — Custom CNN Architecture

The core of the project is a custom **Convolutional Neural Network**.

The architecture follows this general structure:

```text
Input Image
128 × 128 × 3
        ↓
Conv2D
3 → 16 channels
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Max Pooling
        ↓
Conv2D
16 → 32 channels
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Max Pooling
        ↓
Conv2D
32 → 64 channels
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Max Pooling
        ↓
Additional Max Pooling
        ↓
Flatten
        ↓
Fully Connected Layer
        ↓
2 Output Classes
```

---

# 🔬 CNN Layer Breakdown

## Convolution Layer 1

```python
nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
```

Input:

```
3 × 128 × 128
```

The first convolution layer generates **16 feature maps**.

It learns relatively simple image patterns such as:

* Edges
* Lines
* Color boundaries
* Basic facial structures

After max pooling:

```
16 × 64 × 64
```

---

## Convolution Layer 2

```python
nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
```

The number of feature maps increases from:

```
16 → 32
```

After max pooling:

```
32 × 32 × 32
```

At this stage, the CNN can begin learning more complex combinations of low-level features.

---

## Convolution Layer 3

```python
nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
```

Feature maps increase to:

```text
64
```

After max pooling:

```text
64 × 16 × 16
```

These deeper layers can represent more complex visual patterns relevant to distinguishing masked and unmasked faces.

---

# 🔹 Batch Normalization

Each convolution block includes:

```python
nn.BatchNorm2d()
```

Batch normalization helps stabilize intermediate activations during training and can improve optimization behavior.

---

# 🔹 ReLU Activation

The model uses:

```python
nn.ReLU(inplace=True)
```

ReLU introduces non-linearity, allowing the neural network to learn complex relationships within the image data.

---

# 🔹 Max Pooling

Max pooling progressively reduces spatial dimensions.

For example:

```text
128 × 128
     ↓
64 × 64
     ↓
32 × 32
     ↓
16 × 16
     ↓
4 × 4
```

This reduces computational complexity while preserving important learned features.

---

# 🔹 Flattening

Before passing the extracted CNN features into the final classifier, the tensor is flattened.

```python
nn.Flatten()
```

The resulting feature vector contains:

```
64 × 4 × 4
```

values.

---

# 🎯 Output Layer

The final layer is:

```python
nn.Linear(64 * 4 * 4, 2)
```

The two output units correspond to the two classes:

```
With Mask
Without Mask
```

---

# 📉 Step 12 — Loss Function

The project uses:

```python
F.cross_entropy()
```

Cross-entropy loss is used to measure the difference between the model's predicted class outputs and the true labels during training and validation.

The optimization process attempts to minimize this loss.

---

# ⚙️ Step 13 — Model Training

A custom `fit()` function manages the complete training process.

The training configuration in the notebook is:

```python
epochs = 10
grad_clip = 0.1
weight_decay = 1e-4
max_lr = 0.05
```

The optimizer is:

```python
torch.optim.Adam
```

---

# 📈 One Cycle Learning Rate Policy

The project uses:

```python
torch.optim.lr_scheduler.OneCycleLR
```

Instead of maintaining a fixed learning rate throughout training, the learning rate is adjusted dynamically.

The training pipeline therefore becomes:

```
Forward Pass
     ↓
Calculate Loss
     ↓
Backward Propagation
     ↓
Gradient Clipping
     ↓
Adam Optimizer
     ↓
Update Model Weights
     ↓
OneCycleLR Update
     ↓
Validation
```

---

# ✂️ Gradient Clipping

Gradient clipping is applied using:

```python
nn.utils.clip_grad_value_()
```

with:

```python
grad_clip = 0.1
```

This limits large gradient values and can help stabilize optimization.

---

# ⚖️ Weight Decay

The training configuration includes:

```python
weight_decay = 1e-4
```

Weight decay acts as a regularization technique and can help reduce excessive growth of model parameters during optimization.

---

# 📊 Step 14 — Initial Model Evaluation

Before training, the notebook evaluates the randomly initialized model.

The recorded initial validation results are approximately:

```text
Validation Loss: 0.6962
Validation Accuracy: 0.4498
```

This provides a baseline before the CNN learns from the training data.

---

# 🚀 Step 15 — Training Results

The notebook begins training the CNN and records the following first-epoch result:

```text
Epoch [0]
Training Loss: 0.3976
Validation Loss: 0.3368
Validation Accuracy: 0.9732
```

This corresponds to approximately:

```text
97.32% validation accuracy
```

after the recorded first epoch.

> **Note:** The uploaded notebook only contains the recorded output for the first training epoch, so this README does not claim a final 10-epoch accuracy that is not present in the notebook.

The improvement from the initial validation accuracy demonstrates that the CNN learned useful patterns from the training images.

---

# 🧪 Step 16 — Testing & Prediction

A prediction function is created:

```python
def test(images, model):
    out = model(images)
    _, preds = torch.max(out, dim=1)
    return preds.item()
```

The function takes an image tensor, performs inference using the trained model and returns the predicted class.

Conceptually:

```text
New Face Image
      ↓
Preprocessing
      ↓
CNN
      ↓
Feature Extraction
      ↓
Classification
      ↓
With Mask / Without Mask
```

---

# 📊 Model Performance Summary

| Metric                               |      Value |
| ------------------------------------ | ---------: |
| Initial Validation Loss              |     0.6962 |
| Initial Validation Accuracy          |     44.98% |
| Recorded Epoch 0 Training Loss       |     0.3976 |
| Recorded Epoch 0 Validation Loss     |     0.3368 |
| Recorded Epoch 0 Validation Accuracy | **97.32%** |
| Planned Training Epochs              |         10 |
| Batch Size                           |         64 |
| Maximum Learning Rate                |       0.05 |
| Gradient Clip                        |        0.1 |
| Weight Decay                         |       1e-4 |
| Optimizer                            |       Adam |
| LR Scheduler                         | OneCycleLR |

---

# 🗂️ Suggested GitHub Repository Structure

```
Face-Mask-Detection-CNN-PyTorch/
│
├── Face_Mask_Detection.ipynb
│
├── README.md
├── requirements.txt
│
├── images/
│   ├── sample_images.png
│   ├── training_results.png
│   └── prediction_example.png
│
└── .gitignore
```

The dataset itself does not need to be uploaded to GitHub because it can be obtained separately from Kaggle.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Face-Mask-Detection-CNN-PyTorch
```

Install the required packages:

```bash
pip install torch torchvision matplotlib pillow
```

Then launch:

```bash
jupyter notebook
```

Open the project notebook and update the dataset paths if your local directory structure differs from the Kaggle environment used in the original notebook.

---

# 💡 Key Concepts Demonstrated

This project demonstrates several important Deep Learning and Computer Vision concepts:

### Computer Vision

Working with real-world facial images and preparing them for image classification.

### CNN Architecture

Building convolution, batch-normalization, activation, pooling and fully connected layers from scratch.

### Data Augmentation

Using transformations such as horizontal flipping and color jitter to increase training variation.

### Image Normalization

Standardizing image tensors before feeding them into the CNN.

### Batch Processing

Using PyTorch DataLoaders for efficient mini-batch training.

### GPU Acceleration

Automatically using CUDA-capable GPUs when available.

### Regularization

Using weight decay and data augmentation to improve model generalization.

### Gradient Clipping

Controlling large gradient values during optimization.

### Learning Rate Scheduling

Using OneCycleLR to dynamically modify the learning rate during training.

### Model Evaluation

Tracking validation loss and classification accuracy.

---

# 🌍 Real-World Applications

A face-mask classification model could serve as one component of systems used for:

* Workplace safety monitoring
* Public-space monitoring
* Hospital or healthcare facility access systems
* Smart surveillance applications
* Automated compliance analysis
* Computer vision research
* Entrance monitoring systems

For a complete real-time system, this classifier would typically be combined with a **face detection model** to first locate faces in camera frames before classifying each detected face.

---

# 🚀 Future Improvements

The project can be expanded in several ways:

* Add real-time webcam inference
* Add automatic face detection before classification
* Display class names instead of numeric predictions
* Calculate a confusion matrix
* Calculate precision, recall and F1-score
* Plot training vs. validation loss
* Plot validation accuracy across epochs
* Save and reload trained model weights
* Add inference for user-uploaded images
* Compare the custom CNN against transfer-learning architectures
* Experiment with ResNet, MobileNet, or EfficientNet
* Deploy the model using Flask, FastAPI, or Streamlit
* Package the model as a REST API
* Deploy the application to a cloud environment

---

# 🔮 Possible Real-Time Extension

A future version of the project could follow this architecture:

```
Webcam / CCTV Feed
        ↓
Face Detection
        ↓
Extract Face Region
        ↓
Resize & Normalize
        ↓
Trained CNN
        ↓
Mask Classification
        ↓
┌───────────────────┐
│   😷 With Mask    │
│        OR         │
│ 🙂 Without Mask   │
└───────────────────┘
```

This would transform the current image-classification notebook into a complete real-time face-mask monitoring application.

---

# 🎓 What I Learned

Through this project, I gained practical experience in building an end-to-end computer vision classification pipeline using PyTorch.

The project helped strengthen my understanding of:

* Loading image datasets
* Organizing train/validation/test datasets
* Image preprocessing
* Data augmentation
* Tensor transformations
* CNN architecture development
* Convolution operations
* Batch normalization
* Activation functions
* Pooling operations
* Loss calculation
* Backpropagation
* Adam optimization
* Learning-rate scheduling
* Gradient clipping
* GPU/CPU data handling
* Validation and accuracy calculation
* Deep-learning inference

Most importantly, the project demonstrates how raw image data can be transformed into meaningful predictions through a complete deep-learning workflow.

---

# 🏁 Conclusion

This project demonstrates the development of a **Face Mask Detection image classifier using a custom Convolutional Neural Network and PyTorch**.

The complete workflow includes:

```
Dataset Collection
      ↓
Image Preprocessing
      ↓
Data Augmentation
      ↓
Batch Creation
      ↓
CNN Feature Extraction
      ↓
Model Training
      ↓
Validation
      ↓
Testing
      ↓
Mask Classification
```

The recorded training output shows the validation accuracy increasing from an initial **44.98%** to approximately **97.32% after the first recorded training epoch**.

The project provides a strong foundation for understanding practical CNN-based image classification and can be extended into a real-time mask-detection application.

---

## 👨‍💻 Overall Tools

Developed as a **Deep Learning & Computer Vision project using PyTorch**.

**Project:** Face Mask Detection Using CNN & PyTorch
**Domain:** Computer Vision / Deep Learning
**Framework:** PyTorch
**Task:** Binary Image Classification
