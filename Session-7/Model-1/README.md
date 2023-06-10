# Deep Learning Project

## Introduction
This project implements a deep learning model for image classification using the MNIST dataset. The model architecture consists of convolutional and pooling layers followed by fully connected layers. The code also demonstrates the training and evaluation processes, as well as the plotting of training and test metrics.

## Model Architecture
The model architecture is defined in the `Net` class. It consists of several convolutional blocks, each followed by activation, batch normalization, and dropout layers. The final output is passed through a softmax layer for classification.

### Model Summary
|     Layer (type)    |    Output Shape    |  Param # |
|---------------------|--------------------|----------|
|       Conv2d-1      |   [-1, 16, 26, 26] |    144   |
|         ReLU-2      |   [-1, 16, 26, 26] |     0    |
|    BatchNorm2d-3    |   [-1, 16, 26, 26] |    32    |
|       Dropout-4     |   [-1, 16, 26, 26] |     0    |
|       Conv2d-5      |   [-1, 32, 24, 24] |   4,608  |
|         ReLU-6      |   [-1, 32, 24, 24] |     0    |
|    BatchNorm2d-7    |   [-1, 32, 24, 24] |    64    |
|       Dropout-8     |   [-1, 32, 24, 24] |     0    |
|       Conv2d-9      |   [-1, 10, 24, 24] |    320   |
|     MaxPool2d-10    |   [-1, 10, 12, 12] |     0    |
|      Conv2d-11     |   [-1, 16, 10, 10] |   1,440  |
|        ReLU-12     |   [-1, 16, 10, 10] |     0    |
|    BatchNorm2d-13   |   [-1, 16, 10, 10] |    32    |
|      Dropout-14    |   [-1, 16, 10, 10] |     0    |
|      Conv2d-15     |     [-1, 16, 8, 8] |   2,304  |
|        ReLU-16     |     [-1, 16, 8, 8] |     0    |
|    BatchNorm2d-17   |     [-1, 16, 8, 8] |    32    |
|      Dropout-18    |     [-1, 16, 8, 8] |     0    |
|      Conv2d-19     |     [-1, 16, 6, 6] |   2,304  |
|        ReLU-20     |     [-1, 16, 6, 6] |     0    |
|    BatchNorm2d-21   |     [-1, 16, 6, 6] |    32    |
|      Dropout-22    |     [-1, 16, 6, 6] |     0    |
|      Conv2d-23     |     [-1, 16, 6, 6] |   2,304  |
|        ReLU-24     |     [-1, 16, 6, 6] |     0    |
|    BatchNorm2d-25   |     [-1, 16, 6, 6] |    32    |
|      Dropout-26    |     [-1, 16, 6, 6] |     0    |
|      AvgPool2d-27  |     [-1, 16, 1, 1] |     0    |
|      Conv2d-28     |     [-1, 10, 1, 1] |    160   |

Total params: 13,808 
Trainable params: 13,808 
Non-trainable params: 0 


## Calculation of Parameters

Here is the calculation of parameters for each layer:

Conv2d-1: Input shape = (1, 28, 28), Output shape = (16, 26, 26)

Parameters = (input_channels * output_channels * kernel_size * kernel_size) + output_channels = (1 * 16 * 3 * 3) + 16 = 144
Conv2d-5: Input shape = (16, 26, 26), Output shape = (32, 24, 24)

Parameters = (input_channels * output_channels * kernel_size * kernel_size) + output_channels = (16 * 32 * 3 * 3) + 32 = 4,608
Conv2d-9: Input shape = (32, 12, 12), Output shape = (10, 12, 12)

Parameters = (input_channels * output_channels * kernel_size * kernel_size) + output_channels = (32 * 10 * 1 * 1) + 10 = 320
Conv2d-11: Input shape = (10, 12, 12), Output shape = (16, 10, 10)

Parameters = (input_channels * output_channels * kernel_size * kernel_size) + output_channels = (10 * 16 * 3 * 3) + 16 = 1,440
Conv2d-15: Input shape = (16, 10, 10), Output shape = (16, 8, 8)

Parameters = (input_channels * output_channels * kernel_size * kernel_size) + output_channels = (16 * 16 * 3 * 3) + 16 = 2,304
Conv2d-19: Input shape = (16, 8, 8), Output shape = (16, 6, 6)

Parameters = (input_channels * output_channels * kernel_size * kernel_size) + output_channels = (16 * 16 * 3 * 3) + 16 = 2,304
Conv2d-23: Input shape = (16, 6, 6), Output shape = (16, 6, 6)

Parameters = (input_channels * output_channels * kernel_size * kernel_size) + output_channels = (16 * 16 * 3 * 3) + 16 = 2,304
Conv2d-28: Input shape = (16, 1, 1), Output shape = (10, 1, 1)

Parameters = (input_channels * output_channels * kernel_size * kernel_size) + output_channels = (16 * 10 * 1 * 1) + 10 = 160
Total Parameters: 13,808


## Important Code Parts
1. The `train_transforms` and `test_transforms` define the data transformations to be applied during the training and test phases, respectively.
2. The `Net` class defines the model architecture and the forward pass method.
3. The `train` function performs the training loop, including calculating the loss, backpropagation, and updating the model parameters.
4. The `test` function evaluates the model on the test dataset and calculates the test loss and accuracy.
5. The `device` variable is used to determine whether to use CUDA (GPU) or CPU for computation, based on availability.
6. The `summary` function from the `torchsummary` library is used to print the model summary, including the input/output shapes and the number of parameters.
7. The optimizer is initialized with stochastic gradient descent (SGD) as the optimization algorithm with a learning rate and momentum.
8. The training and test losses and accuracies are recorded during the training process for plotting the graphs.
