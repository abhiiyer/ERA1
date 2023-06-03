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

## Calculation of Parameters

| Layer         | Input Size | Output Size | Kernel Size | Parameters |
|---------------|------------|-------------|-------------|------------|
| Conv1         | 1x28x28    | 8x28x28     | 3x3         | 80         |
| Conv2         | 8x28x28    | 16x28x28    | 3x3         | 1168       |
| Pool1         | 16x28x28   | 16x14x14    | 2x2         | 0          |
| Conv3         | 16x14x14   | 16x14x14    | 3x3         | 2320       |
| Conv4         | 16x14x14   | 16x14x14    | 3x3         | 2320       |
| Pool2         | 16x14x14   | 16x7x7      | 2x2         | 0          |
| Conv5         | 16x7x7     | 16x5x5      | 3x3         | 2320       |
| Conv6         | 16x5x5     | 10x3x3      | 3x3         | 1450       |
| GAP           | 10x3x3     | 1x1x10      | -           | 0          |
| **Total**     |            |             |             | **9720**   |

***Please note that the calculation includes the parameters in the convolutional layers, but not in the batch normalization and dropout layers. 
The `summary` function takes all the layers into account and provides an accurate count of the total number of parameters in the network.

The table above shows the calculation of parameters at each layer in the network. It includes the input size, output size, kernel size, and the total number of parameters for each layer.

Total params: 9,720
Trainable params: 9,720
Non-trainable params: 0

The model contains a total of 9,720 parameters, all of which are trainable.

## Calculation of Parameters

### Calculation: (input_channels * output_channels * kernel_height * kernel_width) + output_channels

- **Layer Name: Conv1**
  - Parameters: (1 * 8 * 3 * 3) + 8 = 80

- **Layer Name: Conv2**
  - Parameters: (8 * 16 * 3 * 3) + 16 = 1168

- **Layer Name: Pool1 (MaxPooling)**
  - Calculation: 0 (No parameters in MaxPooling)

- **Layer Name: Conv3**
  - Parameters: (16 * 16 * 3 * 3) + 16 = 2320

- **Layer Name: Conv4**
  - Parameters: (16 * 16 * 3 * 3) + 16 = 2320

- **Layer Name: Pool2 (MaxPooling)**
  - Calculation: 0 (No parameters in MaxPooling)

- **Layer Name: Conv5**
  - Parameters: (16 * 16 * 3 * 3) + 16 = 2320

- **Layer Name: Conv6**
  - Parameters: (16 * 10 * 3 * 3) + 10 = 1450

- **Layer Name: GAP (Global Average Pooling)**
  - Calculation: 0 (No parameters in Global Average Pooling)

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

