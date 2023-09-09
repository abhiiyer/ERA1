
# English to French Translation using Encoder-Decoder Architecture 📚

[![Notebook](https://img.shields.io/badge/Notebook-Open-blue.svg)](https://github.com/abhiiyer/ERA1/blob/main/Session-15/ERA_Session15_v1.ipynb)

## Assignment Overview
The objective of this assignment was to build a model for translating English text to French using the OPUS book translation dataset. The primary goal was to achieve a training loss of less than 1.8.

## Data Source and Preprocessing 📝
- **Dataset:** We used the `OPUS book` translation dataset from Hugging Face.
- **Data Preprocessing:**
  - Implemented dynamic padding to handle sequences of varying lengths efficiently.
  - Removed English sentences longer than 150 characters.
  - Filtered out French sentences longer than the corresponding English sentence by more than 10 characters.

## Model Architecture 🧠
- **Encoder-Decoder:** Our model employs the encoder-decoder architecture for sequence-to-sequence translation tasks.
- **Parameter Sharing:** We utilized parameter sharing with a dense feedforward layer size (dff) set to `1024`.
  - Sharing Pattern: 
    - [e1, e2, e3, e1, e2, e3] - for the encoder
    - [d1, d2, d3, d1, d2, d3] - for the decoder
- **Model Size:** The model comprises approximately 50.6 million parameters.

## Training Configuration ⚙️
- **Batch Size:** We trained the model with a batch size of `32`.
- **Epochs:** The training process ran for `20 epochs`.
- Loss Function: Cross-Entropy

### Learning Rate Scheduler
- **One Cycle Policy Scheduler:** We applied the One Cycle policy scheduler with the following settings:
  - Max learning rate (max_lr): `1e-3`
  - Number of epochs: `20`
  - Percentage of training for the increasing LR phase (pct_start): `10%`
  - Steps per epoch: Determined by the length of the training data loader
  - Learning rate div factor (div_factor): `10`
  - Three-phase learning rate annealing (three_phase)
  - Final learning rate div factor (final_div_factor): `10`
  - Annealing strategy: Linear

- **Optimizer:** We used the ADAM optimizer.

## Training Results 📊
- **Final Loss:** The training loss reached `1.43`.

