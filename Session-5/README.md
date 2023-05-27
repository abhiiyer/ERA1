# MNIST Digit Classification

This project demonstrates the training and evaluation of a convolutional neural network (CNN) model on the MNIST dataset for digit classification. It consists of three main files:

- `model.py`: Defines the structure of the CNN model.
- `utils.py`: Contains utility functions for training and testing the model.
- `S5.ipynb`: Jupyter Notebook file that puts everything together and executes the code.

## Files

### model.py

This file defines the architecture of the CNN model used for digit classification. The `Net` class is derived from `torch.nn.Module` and contains the following layers:

- `conv1`: First convolutional layer with 1 input channel and 32 output channels.
- `conv2`: Second convolutional layer with 32 input channels and 64 output channels.
- `conv3`: Third convolutional layer with 64 input channels and 128 output channels.
- `conv4`: Fourth convolutional layer with 128 input channels and 256 output channels.
- `fc1`: First fully connected layer with 256 * 4 * 4 input features and 50 output features.
- `fc2`: Second fully connected layer with 50 input features and 10 output features (corresponding to the 10 digits).

The `forward` method defines the forward pass of the model, applying convolutional and pooling operations followed by fully connected layers.

### utils.py

This file contains utility functions for training and testing the model.

- `GetCorrectPredCount`: Calculates the number of correct predictions given the predicted values and ground truth labels.
- `train`: Performs the training loop for the model. It takes the model, device, train data loader, optimizer, and criterion as inputs and trains the model for one epoch.
- `test`: Evaluates the model on the test data. It takes the model, device, test data loader, and criterion as inputs and calculates the test loss and accuracy.

### S5.ipynb

This Jupyter Notebook file brings everything together and executes the code. It performs the following steps:

1. Imports the necessary libraries and checks for GPU availability.
2. Defines the train and test data transformations using `torchvision.transforms.Compose`.
3. Loads the MNIST dataset and creates train and test data loaders.
4. Visualizes a batch of images from the training set using matplotlib.
5. Defines the model, optimizer, scheduler, and criterion.
6. Defines empty lists to store the training and testing metrics.
7. Trains the model for a specified number of epochs, recording the training loss and accuracy.
8. Evaluates the model on the test data, recording the test loss and accuracy.
9. Plots the training and test metrics using matplotlib.

## Execution

To execute the code, follow these steps:

1. Make sure you have PyTorch and torchvision installed.
2. Save the `model.py`, `utils.py`, and `S5.ipynb` files in the same directory.
3. Open `S5.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the code cells sequentially.

## Results

The code trains a CNN model on the MNIST dataset and evaluates its performance on the test set. It plots the training and test losses as well as the training and test accuracies.

By analyzing the results, you can observe the model's learning progress and evaluate its performance in classifying digits.

