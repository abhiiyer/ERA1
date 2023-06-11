# Assignment: CIFAR10 Classification with CNN and Albumentations

## Introduction

The goal of this assignment is to design a Convolutional Neural Network (CNN) using PyTorch and the Albumentations library to achieve an accuracy of 85% on the CIFAR10 dataset. The code for this assignment is provided in a Jupyter Notebook, which can be found [here](./EVA8_S6_CIFAR10.ipynb).

The CIFAR10 dataset consists of 60,000 32x32 color training images and 10,000 test images, labeled into 10 classes. The dataset is divided into 50,000 training images and 10,000 validation images.

## Model Architecture

The model architecture used for this assignment is based on the C1C2C3C40 architecture with several modifications. Instead of max pooling, the network consists of 3 convolutional layers with 3x3 filters and a stride of 2. The final layer utilizes global average pooling (GAP). The architecture leverages mobileNetV2, which combines expand, depthwise, and pointwise convolution with residual connections. One layer uses depthwise separable convolution, while another layer uses dilated convolution.

## Data Augmentation

Data augmentation is an essential technique to improve the model's performance and generalization. In this assignment, data augmentation is performed using the Albumentations library. The following techniques are applied in the training data loader:

- Horizontal flipping: Images are horizontally flipped with a certain probability to create additional training samples and introduce more diversity.
- ShiftScaleRotate: Images are randomly shifted, scaled, and rotated to simulate different viewpoints and variations in the dataset.
- CoarseDropout: Random rectangular patches are cut out from the images, which helps the model learn to be robust to missing parts.

Here is a preview of augmented images:

![augmentation](./Images/dataloader_preview.png)

## Results

The model was trained for 25 epochs and achieved an accuracy of 84.64% on the test set. The total number of parameters in the model was under 200k. The training logs, as well as the output of torchsummary, are included in the provided notebook.

Training accuracy: 82.84%
Test accuracy: 84.64%

## Class-wise Accuracy

The class-wise accuracy of the model on the test set is visualized in the following image:

![classwise_accuracy](./Images/classwise_accuracy.png)

## Misclassified Images

Here are a few samples of misclassified images:

![misclassified](./Images/misclassified_images.png)
