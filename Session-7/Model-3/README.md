# MNIST Classification with PyTorch

This repository contains a PyTorch implementation of a convolutional neural network (CNN) for MNIST digit classification. The model achieves a test accuracy of over 99.4% in less than 15 epochs.

## Dataset

The MNIST dataset consists of grayscale images of handwritten digits (0-9) with a resolution of 28x28 pixels. The dataset is divided into a training set and a test set.

## Model Architecture

The model architecture is as follows:

| Layer            | Input Size       | Output Size      | Number of Parameters |
|------------------|------------------|------------------|----------------------|
| Conv2d           | (1, 28, 28)      | (14, 26, 26)     | (1 * 14 * 3 * 3) + 14 = 140               |
| ReLU             | (14, 26, 26)     | (14, 26, 26)     | -                    |
| BatchNorm2d      | (14, 26, 26)     | (14, 26, 26)     | 14                   |
| Dropout          | -                | -                | -                    |
| Conv2d           | (14, 26, 26)     | (30, 24, 24)     | (14 * 30 * 3 * 3) + 30 = 3780             |
| ReLU             | (30, 24, 24)     | (30, 24, 24)     | -                    |
| BatchNorm2d      | (30, 24, 24)     | (30, 24, 24)     | 30                   |
| Dropout          | -                | -                | -                    |
| Conv2d           | (30, 24, 24)     | (10, 24, 24)     | (30 * 10 * 1 * 1) + 10 = 310              |
| MaxPool2d        | (10, 24, 24)     | (10, 12, 12)     | -                    |
| Conv2d           | (10, 12, 12)     | (14, 10, 10)     | (10 * 14 * 3 * 3) + 14 = 1414             |
| ReLU             | (14, 10, 10)     | (14, 10, 10)     | -                    |
| BatchNorm2d      | (14, 10, 10)     | (14, 10, 10)     | 14                   |
| Dropout          | -                | -                | -                    |
| Conv2d           | (14, 10, 10)     | (15, 8, 8)       | (14 * 15 * 3 * 3) + 15 = 1890             |
| ReLU             | (15, 8, 8)       | (15, 8, 8)       | -                    |
| BatchNorm2d      | (15, 8, 8)       | (15, 8, 8)       | 15                   |
| Dropout          | -                | -                | -                    |
| Conv2d           | (15, 8, 8)       | (15, 6, 6)       | (15 * 15 * 3 * 3) + 15 = 2025             |
| ReLU             | (15, 6, 6)       | (15, 6, 6)       | -                    |
| BatchNorm2d      | (15, 6, 6)       | (15, 6, 6)       | 15                   |
| Dropout          | -                | -                | -                    |
| AvgPool2d        | (15, 6, 6)       | (15, 1, 1)       | -                    |
| Conv2d           | (15, 1, 1)       | (15, 1, 1)       | (15 * 15 * 1 * 1) + 15 = 240              |
| BatchNorm2d      | (15, 1, 1)       | (15, 1, 1)       | 15                   |
| ReLU             | (15, 1, 1)       | (15, 1, 1)       | -                    |
| Dropout          | -                | -                | -                    |
| Conv2d           | (15, 1, 1)       | (10, 1, 1)       | (15 * 10 * 1 * 1) + 10 = 160              |

Total number of parameters: 9,962

The model consists of convolutional layers, ReLU activation functions, batch normalization layers, dropout layers, and max pooling layers. The final output is passed through a softmax activation function to obtain the predicted probabilities for each class.

## Training and Testing Process

The training and testing process is as follows:

1. The model is trained using the training dataset, consisting of 60,000 images of handwritten digits.
2. The training process is performed for a specified number of epochs (15 in this case).
3. During each epoch, the model is trained on batches of images and their corresponding labels.
4. The model parameters are updated using the backpropagation algorithm and the chosen optimizer (Stochastic Gradient Descent - SGD).
5. The training loss and accuracy are recorded for each batch and epoch.
6. After training, the model is evaluated using the test dataset, consisting of 10,000 images of handwritten digits.
7. The test accuracy is calculated by comparing the model's predictions with the true labels from the test dataset.

## Dropout

Dropout is a regularization technique used in neural networks to prevent overfitting. It randomly sets a fraction of input units to 0 at each update during training. This forces the network to learn more robust features and reduces the reliance on any single unit. In this model, dropout is applied after each convolutional block.

## Batch Normalization

Batch normalization is a technique used to improve the training of deep neural networks. It normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation. This helps in reducing the internal covariate shift and provides regularization effects. Batch normalization is applied after each convolutional layer in this model.

## What's Right/Wrong with the Model Architecture

The model architecture follows a standard CNN design with convolutional layers, activation functions, pooling layers, batch normalization, and dropout. Overall, the architecture seems reasonable and is capable of achieving high accuracy on the MNIST dataset.

However, there are a few areas where improvements can be made:

1. The model does not include any fully connected layers. This limits its ability to capture high-level features and relationships in the data. Adding fully connected layers after the convolutional layers could potentially improve performance.
2. The model does not incorporate any advanced architectural components like residual connections or skip connections, which have shown to be effective in deep neural networks. Adding these components could help in better feature propagation and gradient flow.
3. The model could benefit from additional regularization techniques such as weight decay or learning rate scheduling to further improve generalization and prevent overfitting.
