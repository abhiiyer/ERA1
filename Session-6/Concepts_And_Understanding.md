# MNIST Classification using PyTorch

This repository contains an implementation of a convolutional neural network (CNN) for MNIST digit classification using PyTorch. The model architecture consists of convolutional layers, batch normalization, dropout, global average pooling, and fully-connected layers. Below are some key concepts and their implementation in PyTorch:

## MaxPooling

MaxPooling is a down-sampling operation used in CNNs to reduce spatial dimensions and extract the most important features. It helps in reducing the computational complexity and providing translation invariance. In PyTorch, MaxPooling is implemented using `nn.MaxPool2d` function.

## 1x1 Convolutions

1x1 Convolutions, also known as pointwise convolutions, are used to change the depth or number of channels in a feature map without affecting its spatial dimensions. It helps in reducing the computational cost and model complexity. In PyTorch, 1x1 Convolutions are implemented using `nn.Conv2d` with a kernel size of 1.

## 3x3 Convolutions

3x3 Convolutions are widely used in CNN architectures as they capture spatial dependencies and are computationally efficient. Multiple stacked 3x3 convolutions can approximate the receptive field of larger kernels while maintaining a lower number of parameters. In PyTorch, 3x3 Convolutions are implemented using `nn.Conv2d` with a kernel size of 3.

## Receptive Field

The receptive field refers to the region of the input space that a particular layer's activations are dependent on. It determines the extent to which a layer can "see" information from the input. In CNNs, the receptive field grows with each layer through successive convolutions and pooling operations.

## SoftMax

SoftMax is an activation function used in the output layer of a classification model. It converts the output values into probabilities, allowing us to interpret the output as the predicted class probabilities. In PyTorch, SoftMax is implemented using `F.log_softmax` in combination with the Negative Log-Likelihood Loss (`nn.CrossEntropyLoss`).

## Learning Rate

The learning rate determines the step size at which the optimizer updates the model's parameters during training. It plays a crucial role in controlling the convergence and performance of the model. A higher learning rate may lead to faster convergence, but it can also result in overshooting. PyTorch provides different optimizers (`optim.SGD`, `optim.Adam`, etc.) that allow specifying the learning rate.

## Kernels and how do we decide the number of kernels?

Kernels are small filters that convolve with the input to extract specific features. The number of kernels determines the number of output channels in a convolutional layer. Deciding the number of kernels is a hyperparameter choice based on the complexity of the problem and the desired model capacity. It is often determined through experimentation and can vary depending on the architecture and dataset.

## Batch Normalization

Batch Normalization is a technique used to normalize the input to a layer by adjusting and scaling the activations. It helps in stabilizing the training process, reducing internal covariate shift, and improving generalization. In PyTorch, Batch Normalization is implemented using `nn.BatchNorm2d` after the convolutional layers.

## Image Normalization

Image Normalization is a preprocessing step that scales the pixel values of an image to a standard range, usually between 0 and 1 or -1 and 1. It helps in improving the convergence of the model during training. In PyTorch, image normalization can be applied using the `transforms.Normalize` function.

## Position of MaxPooling

MaxPooling is typically applied after convolutional layers to downsample the feature maps, reducing spatial dimensions while retaining the most important features. It helps in capturing the most dominant features and provides some degree of translational invariance.

## Concept of Transition Layers

Transition Layers are used to gradually reduce the spatial dimensions and number of channels between two dense blocks in a CNN architecture. They typically consist of a combination of convolutional, pooling, and 1x1 convolutional layers. Transition Layers help in reducing the computational complexity and improving the efficiency of the model.

## Position of Transition Layer

Transition Layers are commonly placed after a set of convolutional layers and before the next dense block. Their position depends on the specific architecture and the desired reduction in spatial dimensions and number of channels.

## DropOut

Dropout is a regularization technique used during training to prevent overfitting. It randomly sets a fraction of the input units to zero at each update, which helps in reducing the co-adaptation of neurons and encourages the model to learn more robust features. In PyTorch, Dropout is implemented using `nn.Dropout` and can be applied after convolutional or fully-connected layers.

## When do we introduce DropOut, or when do we know we have some overfitting

Dropout is introduced during training when there is a risk of overfitting, which typically occurs when the model starts to memorize the training data instead of learning generalizable features. Overfitting can be identified by observing a large gap between the training and validation/test accuracies. Dropout helps in regularizing the model and reducing overfitting by preventing excessive reliance on specific features or neurons.

## The distance of MaxPooling from Prediction

The distance of MaxPooling layers from the prediction layers can vary depending on the specific architecture. In general, MaxPooling is applied to downsample the feature maps and capture the most important features at lower spatial resolutions. The distance between MaxPooling and the prediction layers can affect the receptive field, spatial information, and the level of abstraction in the learned features.

## The distance of Batch Normalization from Prediction

Batch Normalization can be applied at different positions within the model, but it is commonly placed before or after the activation function. The distance of Batch Normalization from the prediction layers can impact the normalization of the activations and the stability of the training process. Placing it closer to the prediction layers allows the model to learn more complex features and improve generalization.

## When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)

The decision to stop convolutions and use a larger kernel or explore other alternatives depends on the specific problem, dataset, and the complexity of the features to be learned. Larger kernels can capture more global or context-dependent information, but they also increase the model's capacity and computational complexity. It is often determined through experimentation and can be based on the performance, convergence, and computational constraints.

## How do we know our network is not going well, comparatively, very early

Early signs that a network is not performing well can be observed by monitoring the training and validation/test metrics, such as accuracy and loss. If the model shows poor convergence, consistently low accuracy, or high loss in the early epochs, it indicates that the network configuration, architecture, or hyperparameters need adjustment. Regular monitoring and evaluation of the training process can help identify and address potential issues early on.

## Batch Size, and Effects of batch size

Batch Size refers to the number of training examples processed together in a single forward and backward pass during training. It affects the model's training dynamics, memory consumption, and computation time. A larger batch size can lead to more stable gradients and faster convergence, but it requires more memory. However, very large batch sizes may result in poorer generalization. The optimal batch size depends on the specific dataset, model complexity, available resources, and computational constraints.

