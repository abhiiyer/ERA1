# CIFAR-10 Classification using ResNet

## Model

The model used for CIFAR-10 classification is ResNet (Residual Neural Network). ResNet is a deep convolutional neural network architecture that is known for its ability to train very deep networks by using skip connections or residual connections. The ResNet model used in this project has a total of 18 convolutional layers, including residual blocks with two convolutional layers each. It also includes batch normalization and ReLU activation functions after each convolutional layer.

## Data Augmentation

Data augmentation is a technique used to increase the diversity of the training dataset by applying random transformations to the input images. It helps to improve the model's ability to generalize and reduces overfitting. The following data augmentation strategies were used:

- Random Crop: Randomly crops the image to a smaller size while preserving the aspect ratio.
- Random Horizontal Flip: Randomly flips the image horizontally.
- Normalization: Normalizes the image by subtracting the mean and dividing by the standard deviation of the dataset.

## Model Parameters & Receptive Field

- Number of epochs: 24
- Batch size: 512

The model's parameters were optimized using the Adam optimizer with a learning rate range obtained from the LR Finder. The learning rate was adjusted using the OneCycleLR scheduler.

The receptive field of a neural network refers to the region of the input image that influences the output of a particular neuron. In this ResNet model, the receptive field increases gradually with the depth of the network due to the stacking of convolutional layers and the use of residual connections.

The ResNet model used in this project consists of multiple residual blocks. Each residual block contains several convolutional layers, batch normalization layers, and ReLU activation functions. The number of filters in each convolutional layer and the size of the receptive field increase as we go deeper into the network. The detailed model architecture and the receptive field size for each layer are as follows:

| Layer                  | Output Size    | Receptive Field | Parameters  |
|------------------------|----------------|-----------------|-------------|
| Input                  | 32x32x3        | -               | -           |
| Convolution 1          | 32x32x64       | 3x3             | 1,792       |
| Residual Block 1       | 32x32x64       | 3x3             | -           |
| Convolution 2          | 32x32x64       | 3x3             | 36,928      |
| Convolution 3          | 32x32x64       | 3x3             | 36,928      |
| Residual Block 2       | 16x16x128      | 5x5             | -           |
| Convolution 4          | 16x16x128      | 3x3             | 147,584     |
| Convolution 5          | 16x16x128      | 3x3             | 295,040     |
| Residual Block 3       | 8x8x256        | 9x9             | -           |
| Convolution 6          | 8x8x256        | 3x3             | 590,080     |
| Convolution 7          | 8x8x256        | 3x3             | 1,180,160   |
| Residual Block 4       | 4x4x512        | 17x17           | -           |
| Convolution 8          | 4x4x512        | 3x3             | 2,359,808   |
| Convolution 9          | 4x4x512        | 3x3             | 2,359,808   |
| Average Pooling        | 1x1x512        | 32x32           | -           |
| Fully Connected (Output)| 10             | -               | 5,130       |

The total number of parameters in the model is 6,574,090.

The model parameters are learned during the training process and are optimized to minimize the loss function.

## Results

The model is trained on the CIFAR-10 dataset, which consists of 50,000 training images and 10,000 test images belonging to 10 different classes. After training the model for a certain number of epochs, the following results are obtained:

- Training Loss: The average loss on the training set.
- Training Accuracy: The accuracy on the training set.
- Test Loss: The average loss on the test set.
- Test Accuracy: The accuracy on the test set.
- Class-wise Accuracy: The accuracy of the model for each class in the test set.

The model achieved the following results on the CIFAR-10 test dataset:

- **Maximum Test Accuracy: 92.35% at Epoch: 22**

## Loss & Accuracy Plots

The following plots show the training and test loss, as well as the training and test accuracy, across the epochs:

![loss_accuracy](./Images/loss_accuracy.png)


