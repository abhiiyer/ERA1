# MNIST Classification with PyTorch

This repository contains a PyTorch implementation for training a convolutional neural network (CNN) to classify the MNIST dataset. The goal is to achieve a validation accuracy of 99.41% or higher while training for less than 20 epochs.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Matplotlib
- torchsummary

## Model Architecture

The model architecture consists of several convolutional layers, batch normalization, dropout, max pooling, and a global average pooling (GAP) layer. The summary of the model is as follows:

| Layer (type)         | Output Shape    | Param # | Tr. Param # |
|----------------------|-----------------|---------|-------------|
| Conv2d-1             | [None, 8, 28, 28]  | 72    | 72       |
| ReLU-2               | [None, 8, 28, 28]  | 0     | 0        |
| BatchNorm2d-3        | [None, 8, 28, 28]  | 16    | 16       |
| Dropout2d-4          | [None, 8, 28, 28]  | 0     | 0        |
| Conv2d-5             | [None, 16, 28, 28] | 1152  | 1152     |
| ReLU-6               | [None, 16, 28, 28] | 0     | 0        |
| BatchNorm2d-7        | [None, 16, 28, 28] | 32    | 32       |
| Dropout2d-8          | [None, 16, 28, 28] | 0     | 0        |
| MaxPool2d-9          | [None, 16, 14, 14] | 0     | 0        |
| Conv2d-10            | [None, 16, 14, 14] | 2304  | 2304     |
| ReLU-11              | [None, 16, 14, 14] | 0     | 0        |
| BatchNorm2d-12       | [None, 16, 14, 14] | 32    | 32       |
| Dropout2d-13         | [None, 16, 14, 14] | 0     | 0        |
| Conv2d-14            | [None, 16, 14, 14] | 2304  | 2304     |
| ReLU-15              | [None, 16, 14, 14] | 0     | 0        |
| BatchNorm2d-16       | [None, 16, 14, 14] | 32    | 32       |
| Dropout2d-17         | [None, 16, 14, 14] | 0     | 0        |
| MaxPool2d-18         | [None, 16, 7, 7]   | 0     | 0        |
| Conv2d-19            | [None, 16, 5, 5]   | 2304  | 2304     |
| ReLU-20              | [None, 16, 5, 5]   | 0     | 0        |
| BatchNorm2d-21       | [None, 16, 5, 5]   | 32    | 32       |
| Dropout2d-22         | [None, 16, 5, 5]   | 0     | 0        |
| Conv2d-23            | [None, 10, 3, 3]   | 1440  | 1440     |
| AdaptiveAvgPool2d-24 | [None, 10, 1, 1]   | 0     | 0        |

Total params: 9,720
Trainable params: 9,720
Non-trainable params: 0

The model contains a total of 9,720 parameters, all of which are trainable.

## Training

The training script will train the model on the MNIST dataset using the Adam optimizer and Cross-Entropy loss. It will print the training and validation accuracy for each epoch. The training will stop automatically once the validation accuracy reaches 99.41% or higher. At the end of the training, a graph showing the accuracy vs loss for both the training and test sets will be displayed.

## Implementation Details

1. The MNIST dataset is loaded using the `datasets.MNIST` class from torchvision. The training and test datasets are then created using the `DataLoader` class, which provides an iterable over the dataset.

2. The model is initialized as an instance of the `Net` class and moved to the GPU if available.

3. The model summary is printed using the `summary` function from the `torchsummary` package. It displays the layer-wise details of the model architecture, including the input and output shapes, the number of parameters, and the number of trainable parameters.

4. The optimizer is defined as Stochastic Gradient Descent (SGD) with a learning rate of 0.01 and momentum of 0.9. The loss function used is Cross Entropy Loss.

5. The training loop runs for a specified number of epochs. In each epoch, the model is trained on the training dataset and evaluated on the test dataset. The training and validation losses are calculated, and the accuracy is computed for both datasets.

6. The best model weights based on the highest validation accuracy are saved.

7. If the validation accuracy reaches 99.41% or higher, the training is stopped.

8. After training, the accuracy and loss curves are plotted using Matplotlib, showing the trend of accuracy and loss over the epochs.
