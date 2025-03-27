# Introduction

Training the ResNet18 model on CIFAR-10 by applying FedAvg and FLTrust algorithms.
The trainings are just experimental and maybe need fine-tuning of the hyperparameters.
They also have been stopped early when the accuracy of 81% was reached as was the initial requirement of the project.

# Setup
Create a virtual environment and run
```pip -r install requirements.txt```

# Train
For training the FedAvg model run
```python3 train.py```

For training the FLTrust model run
```python3 train_fl_trust.py```

# Results

### FedAvg
Trained for 14 rounds 5 epochs each with early stopping after 81% accuracy on test set. 


### FLTrust


Accuracy for FLTrust on full test set: 81.06%

The model reached accuracy of 80% on the 26 round but continued working until 34 round and reached 81.16% with early stopping as the threshold of 81% was reached.

(Experimented with [5, 3, 1] epochs per round)