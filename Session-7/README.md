# ERA1 Session 7 Assignment

## Problem Statement

1. Your new target is:  
        1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)  
        2. Less than or equal to 15 Epochs  
        3. Less than 10000 Parameters (additional points for doing this in less than 8000 pts)  
2. Do this in exactly 3 steps  
3. Each File must have "target, result, analysis" TEXT block (either at the start or the end)
4. You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct.   
5. Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted.   
6. Explain your 3 steps using these target, results, and analysis with links to your GitHub files (Colab files moved to GitHub).   
7. Keep Receptive field calculations handy for each of your models.   
8. If your GitHub folder structure or file_names are messy, -100.   
9. When ready, attempt SESSION 7 - Assignment Solution  


## Solution, Step 1 [Notebook](https://github.com/abhiiyer/ERA1/blob/main/Session-7/Model-1/ERA_Session7_Model-1.ipynb)

### Target   
- Create a Setup (dataset, data loader, train/test steps and log plots)  
- Defining simple model with Convolution block, GAP, dropout and batch normalization.

### Results
- Parameters: 13,808
- Best Train Accuracy 99.35%
- Best Test Accuracy 99.34%  

### Analysis
- Model with 13.8K parameters is able to reach till 99.34% accuracy in 20 epochs.
- Model is not overfitting as training and test accuracies are closeby. (Main purpose was to try to bridge this gap as much possible)

## Solution, Step 2 [Notebook](https://github.com/abhiiyer/ERA1/blob/main/Session-7/Model-2/ERA_Session7_Model2.ipynb)

### Target   
- Add another layer after the GAP, possibly to capture more features and to improve the model performance.

### Results
- Parameters: 9,962
- Best Train Accuracy 98.85%%  
- Best Test Accuracy 99.31%  

### Analysis
- Model with ~9.6K parameters is able to reach till 99.19% accuracy in 15 epochs.
- Adding layers after GAP doesn't show much improvement. (Accuracy almost same)


## Solution, Step 3 [Notebook](https://github.com/abhiiyer/ERA1/blob/main/Session-7/Model-3/ERA_Session7_Model3.ipynb)

### Target   
- Fine Tune the Transforms, set rotation to -10deg to 10deg
- Usage of OneCycleLR Scheduler

### Results
- Parameters: 9,962
- Best Train Accuracy 96.39%  
- Best Test Accuracy 98.80%  

### Analysis
- Model with 9.9K parameters & test 98.80% accuracy in 15 epochs.
- Model does not meets all the requirement of accuracy(<99.4%) and model size (>8K)


## Solution, Step 4 [Notebook](https://github.com/abhiiyer/ERA1/blob/main/Session-7/Model-4/ERA_Session7_Model4.ipynb)

### Target   
- Fine Tune the Transforms, set rotation to -7deg to 7deg
- Usage of StepLR Scheduler

### Results
- Parameters: 7,416
- Best Train Accuracy 99.06%  
- Best Test Accuracy 99.45%  

### Analysis
- Model with 7.4K parameters & test 99.45% accuracy in 15 epochs.
- Model meets all the requirement of model size, test accuracy and epoch.
