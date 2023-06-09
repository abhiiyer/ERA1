# Assignment: CIFAR10 Classification with CNN and Albumentations

## Problem Statement

1. Change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
2. Total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP (compulsory)  - add FC after GAP to target #of classes (optional)
6. use albumentation library and apply:  
        1. horizontal flip
        2. shiftScaleRotate
        3. coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
8. make sure you're following code-modularity (else 0 for full assignment)
9. upload to Github
10. Attempt S9-Assignment Solution.  

## Introduction

The goal of this assignment is to design a Convolutional Neural Network (CNN) using PyTorch and the Albumentations library to achieve an accuracy of 85% on the CIFAR10 dataset. The code for this assignment is provided in a Jupyter Notebook, which can be found [here](./ERA1_S9_CIFAR10.ipynb).

The CIFAR10 dataset consists of 60,000 32x32 color training images and 10,000 test images, labeled into 10 classes. The dataset is divided into 50,000 training images and 10,000 validation images.

# Model Architecture

The model consists of several convolutional layers with different configurations, followed by pooling and fully connected layers.

## Convolutional Block 1

This block consists of two convolutional layers that process the input image.

- Convolutional Layer 1: Applies 32 filters of size 3x3 to the input image. This helps to extract various features from the image.
- Convolutional Layer 2: Applies 64 filters of size 3x3 to the previous feature map. This layer further enhances the learned features.

## Transition Layer 1

This layer reduces the number of channels in the feature map.

- Convolutional Layer: Applies 32 filters of size 1x1 to the previous feature map. It helps in reducing the computational complexity and prepares for the next block.

## Convolutional Block 2

This block consists of multiple depthwise separable convolutions, which are efficient alternatives to standard convolutions.

- Depthwise Separable Convolution: Applies depthwise separable convolution with 32 groups and 3x3 kernel size, followed by 1x1 pointwise convolution. This block is repeated three times. It helps in capturing spatial correlations between channels efficiently.

## Transition Layer 2

This layer reduces the size of the feature map using a larger receptive field.

- Convolutional Layer: Applies 16 filters of size 3x3 with a dilation rate of 2 to the previous feature map. It helps in capturing larger context and spatial information.

## Convolutional Block 3

This block further processes the feature map using depthwise separable convolutions.

- Depthwise Separable Convolution: Applies depthwise separable convolution with 16 groups and 3x3 kernel size, followed by 1x1 pointwise convolution. This block is repeated three times. It helps in extracting more abstract and discriminative features.

## Transition Layer 3

This layer reduces the size of the feature map while preserving important features.

- Convolutional Layer: Applies 64 filters of size 3x3 with a stride of 2 to the previous feature map. It reduces the spatial dimensions of the feature map.
- Convolutional Layer: Applies 16 filters of size 1x1 to the previous feature map. It further reduces the number of channels.

## Convolutional Block 4

This block performs additional convolutions to enhance the learned features.

- Convolutional Layer 1: Applies 32 filters of size 3x3 to the previous feature map.
- Convolutional Layer 2: Applies 64 filters of size 3x3 to the previous feature map.

## Average Pooling

This layer reduces the spatial dimensions of the feature map by taking the average value within each region.

- Applies average pooling with a kernel size of 3x3.

## Convolutional Layer 5

This layer applies 10 filters of size 1x1 to produce the final feature map.

- Convolutional Layer: Applies 10 filters of size 1x1 to the previous feature map. It helps in capturing the class-specific information.

## Output Layer

The output layer processes the final feature map to produce the output probabilities.

- Reshapes the feature map into a 10-dimensional vector.
- Applies the log softmax activation function to produce the output probabilities.

# Pros and Cons

Pros:
- The model utilizes depthwise separable convolutions, which reduce the number of parameters and computational complexity, making the model more efficient.
- The model incorporates various types of convolutions, batch normalization, and dropout, which can help in better feature extraction and regularization.
- The use of ReLU activation functions introduces non-linearity, enabling the model to learn complex patterns and features.
- The model includes average pooling, which can help in spatial dimension reduction and translation invariance.

Cons:
- The model architecture may be complex for beginners to understand due to the presence of different types of convolutions and multiple blocks.
- The model may require longer training time due to the presence of many convolutional layers.
- The effectiveness of the model heavily relies on appropriate hyperparameter tuning and dataset characteristics.

## Data Augmentation

Data augmentation is an essential technique to improve the model's performance and generalization. In this assignment, data augmentation is performed using the Albumentations library. The following techniques are applied in the training data loader:

- Horizontal flipping: Images are horizontally flipped with a certain probability to create additional training samples and introduce more diversity.
- ShiftScaleRotate: Images are randomly shifted, scaled, and rotated to simulate different viewpoints and variations in the dataset.
- CoarseDropout: Random rectangular patches are cut out from the images, which helps the model learn to be robust to missing parts.

Here is a preview of augmented images:

![augmentation](./Images/dataloader_preview.png)

## Model Parameter & Receptive Field

| Layer                  | Output Size   | Receptive Field | Parameters                |
|------------------------|---------------|----------------|---------------------------|
| Conv1                  | 32x32x32      | 3x3            | (3x3x3)x32 = 864          |
| Conv2                  | 32x32x64      | 5x5            | (3x3x32)x64 = 18432       |
| Transition1            | 32x32x32      | 5x5            | (1x1x64)x32 = 2048        |
| Conv2 (Depthwise)      | 32x32x32      | 13x13          | (3x3x32)x32 = 9216        |
| Conv2 (Pointwise)      | 32x32x32      | 13x13          | (1x1x32)x32 = 1024        |
| Conv2 (Depthwise)      | 32x32x32      | 21x21          | (3x3x32)x32 = 9216        |
| Conv2 (Pointwise)      | 32x32x32      | 21x21          | (1x1x32)x32 = 1024        |
| Conv2 (Depthwise)      | 32x32x32      | 29x29          | (3x3x32)x32 = 9216        |
| Conv2 (Pointwise)      | 32x32x32      | 29x29          | (1x1x32)x32 = 1024        |
| Transition2            | 32x32x16      | 37x37          | (3x3x32)x16 = 4608        |
| Conv3 (Depthwise)      | 32x32x16      | 53x53          | (3x3x16)x16 = 2304        |
| Conv3 (Pointwise)      | 32x32x32      | 53x53          | (1x1x16)x32 = 512         |
| Conv3 (Depthwise)      | 32x32x32      | 85x85          | (3x3x32)x32 = 9216        |
| Conv3 (Pointwise)      | 32x32x64      | 85x85          | (1x1x32)x64 = 2048        |
| Conv3 (Depthwise)      | 32x32x64      | 117x117        | (3x3x64)x64 = 36864       |
| Conv3 (Pointwise)      | 32x32x64      | 117x117        | (1x1x64)x64 = 4096        |
| Transition3            | 16x16x16      | 149x149        | (3x3x64)x16 = 36864       |
| Conv4                  | 14x14x64      | 181x181        | (3x3x16)x32 = 4608        |
| Conv4                  | 12x12x64      | 213x213        | (3x3x32)x64 = 18432       |
| AvgPool2D              | 1x1x64        | -              | -                         |
| Conv5                  | 1x1x10        | 213x213        | (1x1x64)x10 = 640         |
| Total Parameters       | -             | -              | 99,968                    |

## Results

The model was trained for 300 epochs and achieved an maximum test accuracy of 85.90% on the test set. The total number of parameters in the model was under 100k. The training logs, as well as the output of torchsummary, are included in the provided notebook.

![test_accuracy](./Images/max_test_accuracy.png)

## Class-wise Accuracy

The class-wise accuracy of the model on the test set is visualized in the following image:

![classwise_accuracy](./Images/class_accuracy.png)

## Loss & Accuracy Plots (Training & Test)

![loss_accuracy](./Images/loss_accuracy.png)
