import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,RandomSampler
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os
import random
from scipy.ndimage import gaussian_filter, median_filter
from matplotlib import pyplot as plt

from pathlib import Path

import time

import scripts.autoencoder_double_tip as ae

# create the datasets
dataset_train = ae.STM_double_tip_dataset(r'/home/uceenlk/Scratch/pytorch/filled_empty/train/')
dataset_test = ae.STM_double_tip_dataset(r'/home/uceenlk/Scratch/pytorch/filled_empty/test/', length = 500)

# create dataloaders and samplers (we want to sample without replacement)
batchSize = 32
sampler_train = RandomSampler(dataset_train, replacement=False)
sampler_test = RandomSampler(dataset_test, replacement=False)
data_loaders_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=batchSize)
data_loaders_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=batchSize)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Create the autoencoder model with no skip connections
autoencoder = ae.Autoencoder()

# Loss and Optimizer
criterion = nn.L1Loss()
optimizer1 = optim.Adam(autoencoder.parameters(), lr=0.001)
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min')

# Train the autoencoder
history1 = ae.train_autoencoder(
    model=autoencoder,
    dataloader_train= data_loaders_train,
    dataloader_test= data_loaders_test,
    loss_fn=criterion,
    optimizer=optimizer1,
    scheduler=scheduler1,
    patience = 25,
    num_epochs=100,
    model_name="best_autoencoder.pth",
    device= "cuda"  
)

np.save("history1_train_loss.npy", history1['train_loss'])
np.save("history1_val_loss.npy", history1['val_loss'])


# Create the autoencoder model with skip connections

UNet_autoencoder = ae.UNetAutoencoder()

# Loss and Optimizer
optimizer2 = optim.Adam(UNet_autoencoder.parameters(), lr=0.001)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min')

# Train the autoencoder
history2 = ae.train_autoencoder(
    model=autoencoder,
    dataloader_train= data_loaders_train,
    dataloader_test= data_loaders_test,
    loss_fn=criterion,
    optimizer=optimizer2,
    scheduler=scheduler2,
    patience = 25,
    num_epochs=100,
    model_name="best_autoencoder_UNet.pth",
    device= "cuda"  
)


np.save("history2_train_loss.npy", history2['train_loss'])
np.save("history2_val_loss.npy", history2['val_loss'])
