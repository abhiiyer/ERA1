# Network Architecture with Parameter Calculation

The code provided defines a convolutional neural network (CNN) architecture and includes the calculation of the number of parameters at each layer, including batch normalization and dropout layers.

## Network Architecture

The network architecture consists of several convolutional blocks and an output block. Each block is composed of convolutional layers, activation functions, normalization layers (batch normalization, group normalization, or layer normalization), and dropout layers.

The architecture can be customized by selecting the normalization method ('BN' for batch normalization, 'GN' for group normalization, or 'LN' for layer normalization) and adjusting the dropout value.

Here is a brief overview of the network architecture:

1. Input Block:
   - Convolutional layer with 8 output channels, kernel size (3, 3), and ReLU activation.
   - Normalization layer (based on the selected method: BN, GN, or LN).
   - Dropout layer with the specified dropout value.

2. Convolution Block 1:
   - Convolutional layer with 16 output channels, kernel size (3, 3), and ReLU activation.
   - Normalization layer.
   - Dropout layer.

3. Transition Block 1:
   - Convolutional layer with 8 output channels and kernel size (1, 1).

4. Pooling Layer:
   - Max pooling layer with kernel size (2, 2).

5. Convolution Block 2:
   - Convolutional layer with 16 output channels, kernel size (3, 3), and ReLU activation.
   - Normalization layer.
   - Dropout layer.

6. Convolution Block 3:
   - Convolutional layer with 8 output channels, kernel size (3, 3), and ReLU activation.
   - Normalization layer.
   - Dropout layer.

7. Convolution Block 4:
   - Convolutional layer with 16 output channels, kernel size (3, 3), and ReLU activation.
   - Normalization layer.
   - Dropout layer.

8. Convolution Block 5:
   - Convolutional layer with 16 output channels, kernel size (3, 3), and ReLU activation.
   - Normalization layer.
   - Dropout layer.

9. Output Block:
   - Global average pooling layer.
   - Convolutional layer with 10 output channels and kernel size (1, 1).

10. Dropout:
    - Dropout layer with the specified dropout value.

## Number of Parameters at Each Layer

The following table presents the number of parameters at each layer, including batch normalization and dropout layers:

| Layer        | Output Shape | # Parameters |
|--------------|--------------|--------------|
| Conv2d_1     | (8, 26, 26)  | 72           |
| BatchNorm2d_1| (8, 26, 26)  | 16           |
| Dropout_1    | (8, 26, 26)  | 0            |
| Conv2d_2     | (16, 24, 24) | 1152         |
| BatchNorm2d_2| (16, 24, 24) | 32           |
| Dropout_2    | (16, 24, 24) | 0            |
| Conv2d_3     | (8, 24, 24)  | 128          |
| MaxPool2d    | (8, 12, 12)  | 0            |
| Conv2d_4     | (16, 10, 10) | 1152         |
| BatchNorm2d_3| (16, 10, 10) | 32           |
| Dropout_3    | (16, 10, 10) | 0            |
| Conv2d_5     | (8, 8, 8)    | 1152         |
| BatchNorm2d_4| (8, 8, 8)    | 16           |
| Dropout_4    | (8, 8, 8)    | 0            |
| Conv2d_6     | (16, 6, 6)   | 1152         |
| BatchNorm2d_5| (16, 6, 6)   | 32           |
| Dropout_5    | (16, 6, 6)   | 0            |
| Conv2d_7     | (16, 4, 4)   | 2304         |
| BatchNorm2d_6| (16, 4, 4)   | 32           |
| Dropout_6    | (16, 4, 4)   | 0            |
| AvgPool2d    | (16, 1, 1)   | 0            |
| Conv2d_8     | (10, 1, 1)   | 160          |
| Dropout_7    | (10, 1, 1)   | 0            |

## Important Code Parts

- `normalizationFx` function: This function returns the normalization layer based on the selected method ('BN', 'GN', or 'LN'). The number of groups for group normalization is determined based on the method chosen.
- `forward` method: This method defines the forward pass of the network, where the input is passed through the defined layers in the specified order.

