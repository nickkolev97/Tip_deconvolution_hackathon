import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os
import random
from scipy.ndimage import gaussian_filter, median_filter
from matplotlib import pyplot as plt
from tqdm import tqdm

from pathlib import Path

import time


########################
## TRANSFORMS
########################
# custom transforms

class Double_tip1(object):
  '''
  Add a double tip artefact to the image. The parameters of the double tip are chosen randomly so
  each image will have a different double tip artefact.
  '''

  def __init__(self):
    pass

  def sigmoid(self, x, a,b):
    '''
    A sigmoid function that takes in an array x and parameters a and b.
    This is used to rescale the pixel values in the "doubled" image (the brighter ones
    are often the ones that are doubled more clearly).
    args:
    x: array of pixel values
    a: parameter of the sigmoid function
    b: parameter of the sigmoid function
    returns:
    array of pixel values rescaled by the sigmoid function
    '''
    return 1 / (1 + np.exp(a-b*x))

  def __call__(self,array):
      # get random values for a and b
      # can adjust limits, these are preliminary
      b = np.random.uniform(8, 10)
      num_stds = np.random.uniform(3, 5)
      a = (torch.mean(array).item() + num_stds*torch.std(array).item()) * b
     # print(a,b,num_stds)
      # Apply the sigmoid function to each pixel value
      sigmoid_array = self.sigmoid(array,a,b)
      # Offset each pixel by a random value in x and y (between 5 and 10)
      # Generate random offsets for x and y
      offset_x = torch.randint(1, 11, (1,))[0]
      offset_y = torch.randint(2, 11, (1,))[0]
      # make negative with 50% chance
      if random.random() < 0.5:
        offset_x = -offset_x
      if random.random() < 0.5:
        offset_y = -offset_y
      #print('offset in x and y: ', offset_x, offset_y)

      # Ensure offsets are within bounds
      _, rows, cols = array.shape

      # 80% of the time, apply median filter to sigmoided image,
      # other 20% apply gaussian
      if random.random() < 0.8:
        # random integer between 0 and 11
        median_size = np.random.randint(1, 7)
        # Apply median filter to the sigmoid image
        filtered_offset = median_filter(sigmoid_array, size=(1,median_size,median_size))
      else:
        # random num between 1 and 10
        gaussian_std = np.random.uniform(1, 10)
        filtered_offset = gaussian_filter(sigmoid_array, sigma=gaussian_std)

      # Create a copy of the base array
      result = torch.clone(array)

      # Add the add_array to the base_array with the offset
      # then crop it so it only includes the parts where the two arrays are overlaid
      if offset_x >= 0 and offset_y >= 0:
        # Positive offsets
        result[:,offset_x:, offset_y:] += filtered_offset[:,:rows-offset_x, :cols-offset_y]
        result = result[:,offset_x:, offset_y:]
        label = array[:,offset_x:, offset_y:]
      elif offset_x >= 0 and offset_y < 0:
        # Negative y-offset
        result[:,offset_x:, :offset_y] += filtered_offset[:,:rows-offset_x, -offset_y:]
        result = result[:,offset_x:, :offset_y]
        label = array[:,offset_x:, :offset_y]
      elif offset_x < 0 and offset_y >= 0:
        # Negative x-offset
        result[:,:offset_x, offset_y:] += filtered_offset[:,-offset_x:, :cols-offset_y]
        result = result[:,:offset_x, offset_y:]
        label = array[:,:offset_x, offset_y:]
      else:  # offset_x < 0 and offset_y < 0
        # Negative offsets for both x and y
        result[:,:offset_x, :offset_y] += filtered_offset[:,-offset_x:, -offset_y:]
        result = result[:,:offset_x, :offset_y]
        label = array[:,:offset_x, :offset_y]

      return torch.stack( (result, label), dim=0)


class RandomScanLineArtefact(object):
    '''
    Randomly adds scan line artefacts to the image.
    Does a few different types of scan line artefacts:
    - Adds a constant to a single line in the image but for a longer length
    - Adds a constant to two lines in the image but for a longer length
    - Adds a sinusoidal wave to a single line in the image
    - Adds a sinusoidal wave to two lines in the image
    - Adds a constant to a single line in the image but for a shorter length
    '''
    def __init__(self, p):
        self.p = p

    def __call__(self, scan):
        r1 = random.random()
        r2 = random.random()
        scan_maxmin = (torch.clone(scan)-torch.min(scan))/(torch.max(scan)-torch.min(scan))

        if r1 < self.p:
          rng = np.random.default_rng(12345) # random number generator
          res = scan.shape[1]
          num_lines = 15 # number of lines to add artefact to
          lines = rng.integers(0,res, (num_lines,)) # which scan lines to augment
          columns = rng.integers(0,res, (num_lines,)) # the columns where the artefacts begin
          lengths = rng.integers(0, int(res*0.8), (num_lines,)) # the length of the artefacts
          add_ons = rng.random(size=(num_lines,))/1.67 # random number between 0 and ~0.6 to add on to a scan line
          # add constant to single line
          for i in range(7):
            scan_maxmin[:, lines[i], columns[i]:columns[i]+lengths[i]] = scan_maxmin[:, lines[i], columns[i]:columns[i]+lengths[i]] + add_ons[i]
          # add constant to two lines one
          for i in range(7,9):
            scan_maxmin[:, lines[i]:lines[i]+2, columns[i]:columns[i]+lengths[i]] = scan_maxmin[:, lines[i]:lines[i]+2, columns[i]:columns[i]+lengths[i]] + add_ons[i]
          # add sinusoidal to single line
          for i in range(9,13):
            end = rng.integers(200,314)/100
            lengths[i] = scan_maxmin[:, lines[i], columns[i]:columns[i]+lengths[i]].shape[0] # correct length in case its too long
            cos = np.cos(np.linspace(0, end, num=lengths[i]) )
            scan_maxmin[:, lines[i], columns[i]:columns[i]+lengths[i]] = scan_maxmin[:, lines[i], columns[i]:columns[i]+lengths[i]] + cos*add_ons[i]
          # add sinusoidal to two lines
          for i in range(13,15):
            end = rng.integers(200,314)/100
            lengths[i] = scan_maxmin[:, lines[i]:lines[i]+2, columns[i]:columns[i]+lengths[i]].shape[1] # correct length in case its too long
            cos = np.cos(np.linspace(0, end, num=lengths[i]) )
            scan_maxmin[:, lines[i]:lines[i]+2, columns[i]:columns[i]+lengths[i]] = scan_maxmin[:, lines[i]:lines[i]+2, columns[i]:columns[i]+lengths[i]] + cos*add_ons[i]

        if r2 < self.p:
          # add some shorter scan line artefacts
          rng = np.random.default_rng(12345) # random number generator
          res = scan.shape[1]
          lines = rng.integers(0,res, (10,)) # which scan lines to augment
          columns = rng.integers(0,res, (10,)) # the columns where the artefacts begin
          lengths = rng.integers(0, int(res*0.1) , (10,)) # the length of the artefacts
          add_ons = rng.random(size=(10,))/1.67 # random number between 0 and ~0.6 to add on to a scan line
          for i in range(10):
            scan_maxmin[:, lines[i], columns[i]:columns[i]+lengths[i]] =scan_maxmin[:, lines[i], columns[i]:columns[i]+lengths[i]] + add_ons[i]

        return scan_maxmin

# Custom transformation for random rotation
class RandomRotation_(object):
    '''
    Randomly rotates the image by an angle chosen from a list of angles.
    '''
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        img.transpose(0,1)
        angle = random.choice(self.angles)
        img_ =  F.rotate(img, angle)
        img_.transpose(0,1)
        return img_


class RandomCreep(object):
    '''
    Adds a creep to the top of the image with a probability of p.
    '''
    def __init__(self, p):
        self.p = p

    def __call__(self, scan):
        r = random.random()
        r2 = random.random()
        if r < self.p:
          # generate 3 random ints
          num_lines = np.random.randint(40,50) # num lines the creep is visible in
          a1 = np.random.randint(0,50)
          a2 = np.random.randint(0,50) # factors in the creep polynomial
          scan_ = torch.clone(scan)
          #print('num_lines: ', num_lines, 'a1: ', a1, 'a2: ', a2, 'r2: ', r2)
          if r2 < 0.5:
            # add creep to top
            if scan.dim() == 3:
              for i in range(0,num_lines):
                j = num_lines-i
                roll = (a1*i*i//250 + a2*i*i//100 +i*i//50) // 10
                scan_[0,2*j:2*(j+1),:] = torch.roll(scan[0,2*j:2*(j+1),:], -roll, dims=1)
            elif scan.dim() == 2:
              for i in range(0,num_lines):
                j = num_lines-i
                roll = a1*i*i//250 + a2*i*i//100 +i*i//50
                scan_[2*j:2*(j+1),:] = torch.roll(scan[2*j:2*(j+1),:], -roll, dims=0)
          else:
            res = scan.shape[1]
            if scan.dim() == 3:
              for i in range(res-num_lines,res):
                j = num_lines-i
                roll = (a1*i*i//250 + a2*i*i//100 +i*i//50) // 10
                scan_[0,2*j:2*(j+1),:] = torch.roll(scan[0,2*j:2*(j+1),:], -roll, dims=1)
            elif scan.dim() == 2:
              for i in range(res-num_lines,res):
                j = num_lines-i
                roll = a1*i*i//250 + a2*i*i//100 +i*i//50
                scan_[2*j:2*(j+1),:] = torch.roll(scan[2*j:2*(j+1),:], -roll, dims=0)
          return scan_

        else:
          return scan


class MedianFilter(object):
    '''
    Applies a median filter to the image.
    '''
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Convert the image to a NumPy array
        img_np = np.array(img)

        # Apply the median filter
        filtered_img_np = median_filter(img_np, size=self.size)

        # Convert the filtered image back to a tensor
        filtered_img = torch.tensor(filtered_img_np)

        return filtered_img


class Get_FFT(object):
    '''
    Applies a 2D FFT to the image.
    '''
    def __init__(self):
        pass

    def __call__(self, img):
        # make a 2d hanning window to avoid edge effects
        hanning = torch.hann_window(img.shape[1])
        hanning = hanning.unsqueeze(0) * hanning.unsqueeze(1)

        # Apply the 2D FFT
        fft_img = torch.fft.fft2(hanning*img[0,:,:])
        # shift the zero to centre
        fft_img = torch.fft.fftshift(fft_img)

        # Compute the magnitude of the FFT
        fft_img_real = torch.real(fft_img)
        fft_img_real = fft_img_real - torch.mean(fft_img_real)
        fft_img_imag = torch.imag(fft_img)
        fft_img_imag = fft_img_imag - torch.mean(fft_img_imag)
        # max/min normalise both of them
        fft_img_real = (fft_img_real - torch.min(fft_img_real))/(torch.max(fft_img_real) - torch.min(fft_img_real))
        fft_img_imag = (fft_img_imag - torch.min(fft_img_imag))/(torch.max(fft_img_imag) - torch.min(fft_img_imag))
       # print('fft_img_real shape: ', fft_img_real.shape)
       # print('fft_img_imag shape: ', fft_img_imag.shape)

        return torch.stack((fft_img_real, fft_img_imag), dim=0)
    
######################################

##################################
## DATASET
##################################

class STM_double_tip_dataset(Dataset):
    
    def __init__(self, image_dir, length = 1500, empty = False):
        '''
        Args:
            image_dir (string): Directory with all the images. These are assumed as being numpy arrays.
                                Either with shape (res,res,2) (filled and empty), or just (res,res) (filled only).
            empty (bool): If True, the images are assumed as having filled and empty state images and both are wanted.
                          If False, we take only the filled state images.
            length (int): size of dataset
        '''
        self.length = length
        self.empty = empty
        self.image_dir = image_dir
        # check all files in image_dir have the same shape
        shapes = [np.load(image_dir+f,allow_pickle=True).shape for f in os.listdir(image_dir)]
        print(shapes)
        if all(len(s)==3 for s in shapes):
            pass
        elif all(len(s)==2 for s in shapes):
            pass
       # else:
       #     file_list = [f for f in os.listdir(image_dir) if len(np.load(image_dir+f,allow_pickle=True).shape)==3]
        else:
            raise ValueError('All files in image_dir must have the same shape. Either (2, res, res) or (res, res). Currently, they are {}'.format(shapes))
        # load the data
        shape = len(shapes[0])
        if empty and shape == 3:
            self.image_files = [torch.tensor(np.load(image_dir+f,allow_pickle=True)) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        elif empty and shape == 2:
            raise ValueError('empty=True only makes sense if the images have two channels.')
        elif not empty and shape == 2:
            self.image_files = [torch.tensor(np.load(image_dir+f,allow_pickle=True)) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        elif not empty and shape == 3:
            self.image_files = [torch.tensor(np.load(image_dir+f,allow_pickle=True))[0,:,:] for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

        self.transforms_both = transforms.Compose([
            Double_tip1(),
            RandomRotation_([0, 90, 180, 270])
            ])

        self.creep_transform = RandomCreep(1)

        self.transforms_noise = transforms.Compose([
            MedianFilter(3),
            #RandomScanLineArtefact(0.5),
        ])

        '''
        Transforms order: random blur, random rotation, random brightness, creep, double tip, crop, resize
        '''

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % len(self.image_files)
        image = self.image_files[idx].unsqueeze(0)
        # max/min normalization
        image = (image-torch.min(image))/(torch.max(image)-torch.min(image))
        #print(image.shape)
        # transforms_both returns the original image (in first channel), and the double tip image in zeroth channel
        image = self.transforms_both(image)
        # We crop it after making the double tip, since the Double_tip1 transform
        # returns a non-square array (due to overlaying two images with a random offset in x and y)
        image = transforms.RandomCrop(256)(image)

        # The "label" is the original image, the "image" is the image with the double tip
        label = image[1,:,:]
        image = image[0,:,:]
       
        # max/min normalization
        image = (image-torch.min(image))/(torch.max(image)-torch.min(image))
        label = (label-torch.min(label))/(torch.max(label)-torch.min(label))


        # Apply noise transforms, these are applied to the image only, not the "label"
      # image = self.transforms_noise(image)

        # get fft
      #  fft = Get_FFT()(image)
        #print('fft shape: ', fft.shape)
        # stack it with the image to get a 3 channel image
      #  image = torch.vstack((image, fft))

        return image.float(), label.float()
        
        # transforms_both returns the original image (in first channel), and the double tip image in zeroth channel
        both = torch.stack((image, label), dim=0)
        image = self.transforms_both(both)


        # separate the image and the label
        label = both[1,:,:]
        image = both[0,:,:]

        # Apply noise transforms, these are applied to the image only, not the label
        image = self.transforms_image_only(image)
        
        return image, label
    

##################################

##################################
## MODELS
##################################

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(nn.Conv2d(1,32, kernel_size=3, padding='same'),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32,32, kernel_size=3, padding='same'),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32,64, kernel_size=3, padding='same'),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(64),
                                     nn.Conv2d(64,64, kernel_size=3, padding='same'),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(64),
                                     nn.MaxPool2d(2,2)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(64,128, kernel_size=3, padding='same'),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(128),
                                         nn.Conv2d(128,128, kernel_size=3, padding='same'),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(128),
        )

        self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                     nn.Conv2d(128, 64, kernel_size=3, padding='same'),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(64),
                                     nn.Conv2d(64, 64, kernel_size=3, padding='same'),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(64),
                                     nn.Conv2d(64, 32, kernel_size=3, padding='same'),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 32, kernel_size=3, padding='same'),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 1, kernel_size=1),
                                     nn.Tanh()
        )

    def forward(self, x):
        # Encoding path
        enc = self.encoder(x)

        # Bottleneck
        bottleneck = self.bottleneck(enc)

        # Decoding path
        output = self.decoder(bottleneck)

        return output
    
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc_relu1 = nn.ReLU(inplace=True)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bottleneck_relu = nn.ReLU(inplace=True)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_relu1 = nn.ReLU(inplace=True)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_relu2 = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoding path
        enc1 = self.enc_relu1(self.enc_conv1(x))
        enc2 = self.enc_relu2(self.enc_conv2(self.pool(enc1)))

        # Bottleneck
        bottleneck = self.bottleneck_relu(self.bottleneck_conv(self.pool(enc2)))

        # Decoding path
        up1 = self.upconv1(bottleneck)
        merge1 = torch.cat([up1, enc2], dim=1)
        dec1 = self.dec_relu1(self.dec_conv1(merge1))
        up2 = self.upconv2(dec1)
        merge2 = torch.cat([up2, enc1], dim=1)          
        dec2 = self.dec_relu2(self.dec_conv2(merge2))  

        # Final output
        out = self.final_conv(dec2)
        return out
    
##################################

##################################
## TRAINING
##################################

def train_autoencoder(model, dataloader_train, dataloader_test, loss_fn, optimizer, scheduler, patience, num_epochs, model_name, device="cuda"):
    """
    Train an autoencoder model and save the best-performing version based on validation loss.

    Args:
        model (torch.nn.Module): The autoencoder model to train.
        dataloader_train (DataLoader): DataLoader for training data.
        dataloader_test (DataLoader): DataLoader for validation data.
        loss_fn (torch.nn.Module): Loss function (e.g., MSELoss).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        patience (int): Number of epochs to wait for improvement in validation loss.
        num_epochs (int): Number of epochs to train.
        model_name (str): Path to save the best model.
        device (str): Device to use for training (e.g., "cuda" or "cpu").

    Returns:
        dict: Training history containing train and validation loss per epoch.
    """
    # Move model to device
    model = model.to(device)
    best_loss = float("inf")  # Track best validation loss
    history = {"train_loss": [], "val_loss": []}
    early_stopping = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        model.train()
        train_loss = 0.0

        # Wrap the dataloader with tqdm for progress bar
        progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)


        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)  # Compare output to input (reconstruction)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            train_loss += loss.item()

        train_loss /= len(dataloader_train)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            # Wrap the dataloader with tqdm for progress bar
            progress_bar_val = tqdm(dataloader_test, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

            for inputs, labels in progress_bar_val:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                # Track loss
                val_loss += loss.item()

        val_loss /= len(dataloader_test)

        # update lr if val_loss unchanged
        scheduler.step(val_loss)

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_name)
            print(f"Best model saved with validation loss: {val_loss:.4f}")
            print("-" * 50)
            early_stopping = 0
        else:
          early_stopping +=1

        if early_stopping > patience:
          print("Early stopping")
          break


        # Print epoch summary
        print("-" * 50)
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print("-" * 50)
        # Store metrics in history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return history

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)