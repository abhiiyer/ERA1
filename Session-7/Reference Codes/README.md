
# Model Evolution

This repository tracks the evolution of a deep learning model for MNIST problem. Each code section represents a different iteration of the model with various modifications and improvements.

## CODE 1: The Setup

**Target:**
- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training & Test Loop

**Results:**
- Parameters: 6.3M
- Best Training Accuracy: 99.99
- Best Test Accuracy: 99.24

**Analysis:**
- Extremely Heavy Model for such a problem
- Model is over-fitting, but we are changing our model in the next step

## CODE 2: The Skeleton

**Target:**
- Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible.
- No fancy stuff

**Results:**
- Parameters: 194k
- Best Train Accuracy: 99.35
- Best Test Accuracy: 99.02

**Analysis:**
- The model is still large, but working.
- We see some over-fitting

## CODE 3: The Lighter Model

**Target:**
- Make the model lighter

**Results:**
- Parameters: 10.7k
- Best Train Accuracy: 99.00
- Best Test Accuracy: 98.98

**Analysis:**
- Good model!
- No over-fitting, model is capable if pushed further

## CODE 4: The Batch Normalization

**Target:**
- Add Batch-norm to increase model efficiency.

**Results:**
- Parameters: 10.9k
- Best Train Accuracy: 99.9
- Best Test Accuracy: 99.3

**Analysis:**
- We have started to see over-fitting now.
- Even if the model is pushed further, it won't be able to get to 99.4

## CODE 5: The Regularization

**Target:**
- Add Regularization, Dropout

**Results:**
- Parameters: 10.9k
- Best Train Accuracy: 99.39 (20th Epoch) & 99.47 (25th)
- Best Train Accuracy: 99.30

**Analysis:**
- Regularization working.
- But with the current capacity, not possible to push it further.
- We are also not using GAP, but depending on a BIG-sized kernel

## CODE 6: The Global Average Pooling

**Target:**
- Add GAP and remove the last BIG kernel.

**Results:**
- Parameters: 6k
- Best Train Accuracy: 99.86
- Best Test Accuracy: 98.13

**Analysis:**
- Adding Global Average Pooling reduces accuracy - WRONG
- We are comparing a 10.9k model with a 6k model. Since we have reduced model capacity, a reduction in performance is expected.

## CODE 7: Increase the Capacity

**Target:**
- Increase model capacity. Add more layers at the end.

**Result:**
- Parameters: 11.9k
- Best Train Accuracy: 99.33
- Best Test Accuracy: 99.04

**Analysis:**
- The model still showing over-fitting, possibly Dropout is not working as expected! Wait yes! We don't know which layer is causing over-fitting.
- Quite Possibly we need to add more capacity, especially at the end.
- Closer analysis of MNIST can also reveal that just at RF of 5x5 we start to see patterns forming.
- We can also increase the capacity of the model by adding a layer after GAP!

## CODE 8: Correct MaxPooling Location

**Target:**
- Increase model capacity at the end (add a layer after GAP)
- Perform MaxPooling at RF=5
- Fix Dropout, add it to each layer

**Results:**
- Parameters: 13.8k
- Best Train Accuracy: 99.39
- Best Test Accuracy: 99.41 (9th Epoch)

**Analysis:**
- Works!
- But we're not seeing 99.4 or more as often as we'd like. We can further improve it.
- The model is not over-fitting at all.
- Seeing image samples, we can see that we can add slight rotation.

## CODE 9: Image Augmentation

**Target:**
- Add rotation, our guess is that 5-7 degrees should be sufficient.

**Results:**
- Parameters: 13.8k
- Best Train Accuracy: 99.15
- Best Test Accuracy: 99.5 (18th Epoch)

**Analysis:**
- The model is under-fitting now. This is fine, as we know we have made our train data harder.
- The test accuracy is also up, which means our test data had a few images which had transformation differences w.r.t. the train dataset
