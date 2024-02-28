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
torch.manual_seed(42)
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
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
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
torch.save(model, 'model_newdv2l.pth')