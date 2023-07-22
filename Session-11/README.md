# ResNet-18 for CIFAR-10 Classification

This repository contains code for training and evaluating a ResNet-18 model on the CIFAR-10 dataset. The main objective is to classify images from the CIFAR-10 dataset into one of ten classes. The trained model is saved and can be used for further evaluation and visualization, including the generation of GradCAM heatmaps for misclassified images.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [GradCAM Visualization](#gradcam-visualization)
6. [Model Summary](#model-summary)
7. [Accuracy and Loss Graphs](#accuracy-and-loss-graphs)
8. [Explanation of GradCAM](#explanation-of-gradcam)
9. [Model Architecture](#model-architecture)
10. [Code Flow](#code-flow)
11. [Pros and Cons of GradCAM](#pros-and-cons-of-gradcam)

## Prerequisites

To run the code, you need the following prerequisites:

- Python 3.x
- PyTorch
- TorchVision
- Matplotlib
- NumPy
- OpenCV

You can install the necessary packages using the following command:

```bash
pip install torch torchvision matplotlib numpy opencv-python


# Setup

To set up the environment, ensure you have installed Python 3.x and the required dependencies. You can install the necessary packages using the provided command.

# Training

The training process can be executed by running the main.ipynb notebook. The notebook contains code for loading the CIFAR-10 dataset, defining the ResNet-18 model, setting hyperparameters, training the model, and saving the trained model as resnet18_cifar10.pth. The training loop consists of multiple epochs, where the model is trained using stochastic gradient descent (SGD) with a learning rate scheduler based on the OneCycleLR policy.

# Evaluation

To evaluate the trained model on the CIFAR-10 test set, run the main.ipynb notebook. It loads the trained model, loads the CIFAR-10 test data, and calculates the test loss and accuracy. The evaluation results will be displayed in the notebook.

# GradCAM Visualization

The GradCAM visualization technique is used to visualize which regions of an image contribute most to the model's prediction. To generate GradCAM heatmaps for misclassified images, run the main.ipynb notebook. The code loads the saved model, identifies misclassified images, and plots the images along with their corresponding GradCAM heatmaps.

# Customization

If you want to experiment with different hyperparameters or model architectures, you can modify the main.ipynb notebook. Additionally, if you wish to change the data augmentation or transformation strategies, you can update the transforms in the data_loader.py file.

# Explanation of GradCAM

GradCAM (Gradient-weighted Class Activation Mapping) is a visualization technique that highlights the regions of an image that are important for the model's prediction. It achieves this by generating a heatmap that shows which pixels in the image contribute most to the model's decision.

The GradCAM technique involves the following steps:

1. Forward pass: The input image is passed through the trained model to get the output logits.

2. Backward pass: The gradients of the output logits with respect to the convolutional feature maps are calculated.

3. Global average pooling: The gradients are spatially averaged to get the importance weights for each feature map.

4. Weighted combination: The feature maps are linearly combined based on their importance weights to obtain the final heatmap.

## Model Architecture

The ResNet-18 model is a deep convolutional neural network consisting of multiple residual blocks. It has 18 layers and can be used for various image classification tasks. The architecture is as follows:



## Code Flow

The code in the main.ipynb notebook follows the following flow:

1. Import the required libraries and modules.

2. Set up the device (GPU or CPU) for training.

3. Define the hyperparameters for training, such as batch size, number of epochs, and learning rate.

4. Load the CIFAR-10 dataset and create data loaders for training and testing.

5. Create the ResNet-18 model and define the loss function and optimizer.

6. Implement the training loop, where the model is trained for the specified number of epochs.

7. Save the trained model to a file.

8. Load the saved model for evaluation.

9. Use GradCAM to generate heatmaps for misclassified images.

## Pros and Cons of GradCAM

### Pros:

- GradCAM is a simple and effective technique for visualizing the decision-making process of deep neural networks.

- It provides insights into which regions of an image the model focuses on to make predictions.

- GradCAM heatmaps can be easily generated and applied to any CNN-based model without additional model modifications.

### Cons:

- GradCAM only highlights regions in the input image but does not provide insights into why the model makes specific predictions.

- It may not be suitable for models with complex decision-making processes, as the heatmaps may not always accurately reflect the model's reasoning.

- GradCAM heatmaps may not be interpretable for highly ambiguous images, where multiple regions contribute equally to the model's decision.
