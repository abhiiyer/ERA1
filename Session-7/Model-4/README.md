# MNIST Classification with PyTorch

This repository contains a PyTorch implementation of a convolutional neural network (CNN) for MNIST digit classification. The model achieves a test accuracy of over 99.4% in less than 15 epochs.

## Dataset

The MNIST dataset consists of grayscale images of handwritten digits (0-9) with a resolution of 28x28 pixels. The dataset is divided into a training set and a test set.

## Model Architecture

The model architecture is as follows:

| Layer            | Input Size       | Output Size      | Number of Parameters |
|------------------|------------------|------------------|----------------------|
| Conv2d           | (1, 28, 28)      | (8, 26, 26)      | 72                   |
| ReLU             | (8, 26, 26)      | (8, 26, 26)      | -                    |
| Conv2d           | (8, 26, 26)      | (16, 24, 24)     | 1,152                |
| ReLU             | (16, 24, 24)     | (16, 24, 24)     | -                    |
| BatchNorm2d      | (16, 24, 24)     | (16, 24, 24)     | 32                   |
| Dropout          | -                | -                | -                    |
| Conv2d           | (16, 24, 24)     | (8, 24, 24)      | 1,152                |
| MaxPool2d        | (8, 24, 24)      | (8, 12, 12)      | -                    |
| Conv2d           | (8, 12, 12)      | (16, 10, 10)     | 1,152                |
| ReLU             | (16, 10, 10)     | (16, 10, 10)     | -                    |
| BatchNorm2d      | (16, 10, 10)     | (16, 10, 10)     | 32                   |
| Dropout          | -                | -                | -                    |
| Conv2d           | (16, 10, 10)     | (8, 8, 8)        | 1,152                |
| ReLU             | (8, 8, 8)        | (8, 8, 8)        | -                    |
| BatchNorm2d      | (8, 8, 8)        | (8, 8, 8)        | 16                   |
| Dropout          | -                | -                | -                    |
| Conv2d           | (8, 8, 8)        | (16, 6, 6)       | 1,152                |
| ReLU             | (16, 6, 6)       | (16, 6, 6)       | -                    |
| BatchNorm2d      | (16, 6, 6)       | (16, 6, 6)       | 32                   |
| Dropout          | -                | -                | -                    |
| Conv2d           | (16, 6, 6)       | (16, 4, 4)       | 2,304                |
| ReLU             | (16, 4, 4)       | (16, 4, 4)       | -                    |
| BatchNorm2d      | (16, 4, 4)       | (16, 4, 4)       | 32                   |
| Dropout          | -                | -                | -                    |
| AvgPool2d        | (16, 4, 4)       | (16, 1, 1)       | -                    |
| Conv2d           | (16, 1, 1)       | (10, 1, 1)       | 160                  |

Total number of parameters: 7,416

The model consists of convolutional blocks, transition blocks, and an output block. ReLU activation functions, batch normalization, and dropout layers are applied to improve the model's performance and generalization.

## Training Process

The training process involves the following steps:

1. The training images are randomly rotated by an angle between -7.0 and 7.0 degrees to increase the diversity of the training data.
2. The training images are normalized to have a mean of 0.1307 and a standard deviation of 0.3081.
3. The model is trained using the stochastic gradient descent (SGD) optimizer with a learning rate of 0.02 and momentum of 0.9.
4. The learning rate is adjusted using the StepLR scheduler, which reduces the learning rate by a factor of 0.1 every 6 epochs.
5. The model is trained for a total of 15 epochs. During each epoch, the model is trained on batches of training data, and the loss and accuracy are recorded.
6. After each epoch, the model is tested on the test set to evaluate its performance. The average loss and accuracy on the test set are calculated and displayed.

## Testing Process

The testing process involves the following steps:

1. The model is set to evaluation mode, which disables the dropout layers.
2. The model is evaluated on the test set by forwarding the test images through the model.
3. The test loss is calculated as the average negative log-likelihood loss.
4. The predictions of the model are compared with the true labels to calculate the test accuracy.
5. The average test loss and accuracy are displayed.

## Optimizer: Stochastic Gradient Descent (SGD)

SGD is an optimization algorithm commonly used for training deep learning models. It updates the model's parameters based on the gradients of the loss function with respect to those parameters. The momentum parameter helps accelerate SGD by accumulating past gradients and smoothing out the updates.

## Learning Rate Scheduler: StepLR

StepLR is a learning rate scheduler that adjusts the learning rate during training. It reduces the learning rate by a specified factor after a certain number of epochs. In this implementation, the learning rate is reduced by a factor of 0.1 every 6 epochs.

Feel free to modify the hyperparameters, such as the learning rate, batch size, or number of epochs, to experiment with different configurations.


## Dependencies

The following dependencies are required to run the code:

- torch
- torchvision
- tqdm

## Instructions

To run the code, follow these steps:

1. Install the required dependencies mentioned above.
2. Execute the code in a Python environment.
3. The code will download the MNIST dataset automatically.
4. The training progress will be displayed in the console, showing the loss, batch index, and accuracy.
5. After the training is completed, the test accuracy will be displayed.

Feel free to modify the hyperparameters, such as the learning rate, batch size, or number of epochs, to experiment with different configurations.
