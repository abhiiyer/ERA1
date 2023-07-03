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

## Results

The model achieved the following results on the CIFAR-10 test dataset:

- Maximum Test Accuracy: [INSERT MAXIMUM TEST ACCURACY] at Epoch: [INSERT EPOCH NUMBER]

## Loss & Accuracy Plots

The following plots show the training and test loss, as well as the training and test accuracy, across the epochs:

![Loss and Accuracy Plots](loss_accuracy_plots.png)


