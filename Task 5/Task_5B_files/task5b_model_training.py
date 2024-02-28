"""
* Team Id: GG_2526
* Author List: Harshal Kale, Akash Mohapatra, Sharanya Anil
* Filename: task6_model_training.py
* Theme: GeoGuide
* Functions: data_setup.create_dataloaders, engine.train, torch.manual_seed, torch.cuda.manual_seed, torch.optim.Adam, nn.CrossEntropyLoss, torch.save
* Global Variables: num_epochs, batch_size, learning_rate, train_data_dir, test_data_dir, classes, weights, auto_transforms, loss_fn, optimizer, start_time, end_time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms

# Try to get torchinfo, install it if it doesn't work
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    !pip install -q torchinfo
    from torchinfo import summary

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
    from going_modular.going_modular import data_setup, engine
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    !git clone https://github.com/mrdbourke/pytorch-deep-learning
    !mv pytorch-deep-learning/going_modular .
    !rm -rf pytorch-deep-learning
    from going_modular.going_modular import data_setup, engine


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Setup num_epochs, batch_size and learning_rate
num_epochs = 30
batch_size = 32
learning_rate = 0.001

# Setup the train and test dataset directory
train_data_dir = 'GG/train'
test_data_dir = 'GG/test'

# Define classes for classifiaction task
classes = ['combat', 'destroyedbuilding', 'fire', 'humanitarianaid', 'militaryvehicles']


# Get a set of pretrained model weights (Efficientnet_v2_l here)
weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
weights


# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()
auto_transforms


'''
* Function Name: create_dataloaders
* Input: train_dir - Directory path containing the training data.
         test_dir - Directory path containing the testing data.
         transform (torchvision.transforms.Compose) - Data transformations to be applied to the images.
         batch_size - Batch size for the DataLoader.
* Output: train_dataloader (torch.utils.data.DataLoader) - DataLoader for the training data.
          test_dataloader (torch.utils.data.DataLoader) - DataLoader for the testing data.
          class_names - List of class names present in the dataset.
* Logic: Creates DataLoader instances for training and testing data using the specified directory paths, data transformations, and batch size.
* Example Call: create_dataloaders(train_dir=train_data_dir, test_dir=test_data_dir, transform=auto_transforms, batch_size=32)
'''

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_data_dir,
                                                                               test_dir=test_data_dir,
                                                                               transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
                                                                               batch_size=32) # set mini-batch size to 32

train_dataloader, test_dataloader, class_names


# Load the efficientnet_v2_l model with default pretrained weights
weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT # .DEFAULT = best available weights 
# Move the model to the specific device (cpu or gpu)
model = torchvision.models.efficientnet_v2_l(weights=weights).to(device)


# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False


# Set the manual seeds
'''
* Function Name: torch.manual_seed
* Input: seed - THe seed value to set for random number generation.
* Output: None
* Logic: Sets the seed for generating random numbers using the CPU.
        This ensures that the random numbers generated during execution are reproducible.
* Example Call: torch.manual_seed(42)
'''
torch.manual_seed(42)


'''
* Function Name: torch.cuda.manual_seed
* Input: seed (int) - The seed value to set for random number generation on the GPU.
* Output: None
* Logic: Sets the seed for generating random numbers using the GPU.
          This ensures that the random numbers generated during execution on the GPU are reproducible.
* Example Call: torch.cuda.manual_seed(42)
'''
torch.cuda.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)


# Define loss and optimizer
'''
* Function Name: nn.CrossEntropyLoss
* Input: Tensors
* Output: loss_fn - Loss function instance.
* Logic: Creates an instance of the Cross Entropy Loss function.
          This loss function is commonly used for multi-class classification problems. 
          It combines softmax activation and negative log likelihood loss.
* Example Call: loss_fn = nn.CrossEntropyLoss()
'''
loss_fn = nn.CrossEntropyLoss()


'''
* Function Name: torch.optim.Adam
* Input: params - Iterable of parameters to optimize or dicts defining parameter groups.
         lr - Learning rate
         betas (Tuple[float, float], optional) - Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
         eps - Term added to the denominator to improve numerical stability (default: 1e-8).
         weight_decay - Weight decay (L2 penalty) (default: 0).
         amsgrad - Whether to use the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond" (default: False).
* Output: optimizer (torch.optim.Adam) - Optimizer instance.
* Logic: Creates an instance of the Adam optimizer, which is an adaptive learning rate optimization algorithm.
          It automatically adjusts the learning rate during training based on the average of past gradients.
          The algorithm computes individual adaptive learning rates for different parameters.
* Example Call: optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
'''
* Function Name: train
* Input: model (torch.nn.Module) - The neural network model to be trained.
         train_dataloader (torch.utils.data.DataLoader) - DataLoader for the training data.
         test_dataloader (torch.utils.data.DataLoader) - DataLoader for the testing data.
         optimizer (torch.optim.Optimizer) - The optimizer used for training.
         loss_fn (torch.nn.Module) - The loss function used for training.
         epochs - Number of training epochs.
         device - Device to run the training on (e.g., "cuda" for GPU, "cpu" for CPU).
* Output: results - Dictionary containing training and testing metrics.
* Logic: Trains the provided model using the specified DataLoader instances, optimizer, loss function, and device.
         Evaluates the model's performance on the testing data after each epoch.
         Returns a dictionary containing training and testing metrics.
* Example Call: engine.train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device)
'''

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=30,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model and its weights
torch.save(model.state_dict(), 'model_newdv2l_weights.pth')


'''
* Function Name: torch.save
* Input: Any object to be saved. This can be a model, tensor, or any other serializable object.
         filepath - Filepath where the object will be saved.
* Output: None
* Logic: Saves the specified object to the specified file path using PyTorch's serialization format.
          This function allows you to save trained models, tensors, or any other serializable objects to disk.
* Example Call: torch.save(model, 'model_newdv2l.pth')
'''
torch.save(model, 'model_newdv2l.pth')