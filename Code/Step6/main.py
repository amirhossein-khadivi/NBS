### import libraries
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import torch.optim as optim
from collections import deque
import numpy as np
import pickle
import sys
import cv2
import re
import gc
import psutil
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

### define adderss dataset
unlabel_train_path = '/content/content/MyDrive/MVTec/bottle/train/good'
unlabel_test_path = '/content/content/MyDrive/MVTec/bottle/test/good'
label_path = '/content/content/MyDrive/MVTec/bottle/test'
ground_truth_path = '/content/content/MyDrive/MVTec/bottle/ground_truth'
autoencoder_path = '/content/content/MyDrive/MVTec/bottle/autoencoder.pth'

# Basic image transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor and scale pixel values to [0, 1]
])

# Data augmentation transformations to increase training data variability
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),              # Flip the image horizontally with a 50% chance
    transforms.RandomVerticalFlip(p=0.5),                # Flip the image vertically with a 50% chance
    transforms.RandomRotation(degrees=30, fill=0),       # Randomly rotate the image up to ±30 degrees, filling empty pixels with 0
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),  # Randomly translate the image by up to 10% in both directions
    transforms.RandomAffine(degrees=0, shear=20, fill=0),              # Apply random shear transformation up to 20 degrees
])


### Sobel filter function with compute R_clone
def filtering(input):
    """
    Compute edge intensity map using Sobel edge detection after applying Gaussian blur
    to grayscale image patches. This is used to highlight edges and textures, encouraging
    the model to focus on regions with structural information.

    Parameters:
        input: Tensor of shape [batch, channels, height, width]
               (image patches, assumed to be RGB)

    Returns:
        edge_intensity: Tensor of shape [batch, 1, height, width]
                        representing edge strength per pixel (R_clone)
    """
    # Define Sobel kernels for x and y gradients (edge detection)
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

    # Convert RGB image to grayscale using luminance-preserving formula
    gray_image = 0.2989 * input[:, 0:1, :, :] + \
                 0.5870 * input[:, 1:2, :, :] + \
                 0.1140 * input[:, 2:3, :, :]

    # Define a 5x5 Gaussian kernel for smoothing (reduces noise before edge detection)
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1,  4,  6,  4, 1]], dtype=torch.float32).view(1, 1, 5, 5) / 256.0

    # Apply Gaussian blur to smooth the grayscale image
    smoothed_image = F.conv2d(gray_image, kernel.to(input.device), padding=2)

    # Apply Sobel filters to compute horizontal and vertical edge gradients
    edges_x = F.conv2d(smoothed_image, sobel_x.to(input.device), padding=1)
    edges_y = F.conv2d(smoothed_image, sobel_y.to(input.device), padding=1)

    # Combine x and y gradients to get overall edge magnitude (Euclidean norm)
    edge_intensity = torch.sqrt(edges_x**2 + edges_y**2)

    return edge_intensity


### Neural Batch Sampler class with patch selection and image traversal functions
class NeuralBatchSampler(nn.Module):
    """
    A neural module for learning to extract informative patches from images using reinforcement learning.
    Predicts movement actions for patches and maintains a visit-history channel.

    Attributes:
        input_size (int): Size of the input image (assumes square images).
        crop_size (int): Size of the region to be cropped from the image.
        patch_size (int): Size of the final extracted patch.
        move_size (int): Number of pixels to move when taking an action.
        action_space (int): Number of possible movement actions (including 'no move').
    """

    def __init__(self, input_size=900, crop_size=128, patch_size=64, move_size=24, action_space=9):
        super(NeuralBatchSampler, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.move_size = move_size
        self.action_space = action_space

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1)  # Input: 6 channels (e.g., RGB + history)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Fully connected layers for action prediction
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, action_space)

        # Batch normalization for each convolutional block
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

    def forward(self, patches):
        """
        Forward pass through the model to predict action probabilities for a given batch of patches.

        Args:
            patches (Tensor): Input tensor of shape [batch_size, 6, H, W].

        Returns:
            Tensor: Softmax probabilities over the action space [batch_size, action_space].
        """
        x = F.relu(self.bn1(self.conv1(patches)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)

        return x

    def select_crops(self, image, center=None):
        """
        Extract a square crop from the input image at the given center.

        Args:
            image (Tensor): Input image tensor of shape [1, C, H, W].
            center (Tensor): Center coordinate [x, y] of the crop. If None, it must be generated externally.

        Returns:
            patch (Tensor): Extracted image patch of shape [1, C, crop_size, crop_size].
            center (Tensor): The possibly adjusted center used for cropping.
        """
        with torch.no_grad():
            _, _, height, width = image.shape

            # Adjust the center to make sure the crop stays within image bounds
            center[0] = int(center[0].item())
            center[1] = int(center[1].item())

            if center[0] - self.crop_size // 2 < 0:
                center[0] += (self.crop_size // 2 - center[0])
            elif center[0] + self.crop_size // 2 > height:
                center[0] -= (center[0] + self.crop_size // 2 - height)

            if center[1] - self.crop_size // 2 < 0:
                center[1] += (self.crop_size // 2 - center[1])
            elif center[1] + self.crop_size // 2 > width:
                center[1] -= (center[1] + self.crop_size // 2 - width)

            x_start = int(center[0] - self.crop_size // 2)
            y_start = int(center[1] - self.crop_size // 2)
            x_end = x_start + self.crop_size
            y_end = y_start + self.crop_size

            patch = image[:, :, y_start:y_end, x_start:x_end]

            return patch, center

    def move_select_patches(self, center, action, image):
        """
        Move a patch center according to the given action, extract the patch, and update history channel.

        Args:
            center (Tensor): Current center [x, y].
            action (int): Action index (0 = no move).
            image (Tensor): Image of shape [1, 6, H, W] (6 channels: RGB + 3 others).

        Returns:
            patch (Tensor): Extracted patch of shape [1, 3, patch_size, patch_size].
            patch_channel_6 (Tensor): Corresponding patch from channel 6.
            new_center (Tensor): Updated center [x, y].
        """
        with torch.no_grad():
            _, _, height, width = image.shape

            # Define (dx, dy) movement per action
            # Define movement directions for each action index:
            # Action 0: No movement (masked out, dx/dy are NaN)
            # Action 1: Move left        (-move_size,  0)
            # Action 2: Move right       (+move_size,  0)
            # Action 3: Move up          (0, -move_size)
            # Action 4: Move down        (0, +move_size)
            # Action 5: Move up-left     (-move_size, -move_size)
            # Action 6: Move down-left   (-move_size, +move_size)
            # Action 7: Move up-right    (+move_size, -move_size)
            # Action 8: Move down-right  (+move_size, +move_size)
            dx = torch.tensor([float('nan'), -self.move_size, self.move_size, 0, 0,
                               -self.move_size, -self.move_size, self.move_size, self.move_size])
            dy = torch.tensor([float('nan'), 0, 0, -self.move_size, self.move_size,
                               -self.move_size, self.move_size, -self.move_size, self.move_size])

            new_x = center[0].clone().float() + dx[action]
            new_y = center[1].clone().float() + dy[action]

            new_x = torch.clamp(new_x, min=self.crop_size // 2, max=width - self.crop_size // 2)
            new_y = torch.clamp(new_y, min=self.crop_size // 2, max=height - self.crop_size // 2)

            new_center = torch.tensor([new_x, new_y])

            x_start = int(new_x) - self.patch_size // 2
            y_start = int(new_y) - self.patch_size // 2
            x_end = x_start + self.patch_size
            y_end = y_start + self.patch_size

            # Extract RGB patch and counter patch
            patch = image[:, :3, y_start:y_end, x_start:x_end]
            patch_channel_6 = image[:, 4, y_start:y_end, x_start:x_end]

            # Update the visit-count layer (channel 5)
            visit_mask = torch.zeros_like(image[:, 4, :, :])
            visit_mask[:, y_start:y_end, x_start:x_end] = 1
            image[:, 4, :, :] += visit_mask

            del visit_mask
            gc.collect()
            torch.cuda.empty_cache()

            return patch, patch_channel_6, new_center

    def initial_patches(self, images):
        """
        Randomly sample initial patch centers and extract patches from a batch of images.

        Args:
            images (Tensor): Tensor of shape [batch_size, 6, H, W]

        Returns:
            patches (Tensor): RGB patches [batch_size, 3, patch_size, patch_size]
            patches_channel_6 (Tensor): Counter patches [batch_size, 1, patch_size, patch_size]
            centers (Tensor): Centers of extracted patches [batch_size, 2]
        """
        with torch.no_grad():
            batch_size, _, height, width = images.shape

            x_centers = torch.randint(self.crop_size // 2, height - self.crop_size // 2, (batch_size,), device=images.device)
            y_centers = torch.randint(self.crop_size // 2, width - self.crop_size // 2, (batch_size,), device=images.device)
            centers = torch.stack([x_centers, y_centers], dim=1)

            patches = torch.zeros((batch_size, 3, self.patch_size, self.patch_size), device=images.device)
            patches_channel_6 = torch.zeros((batch_size, 1, self.patch_size, self.patch_size), device=images.device)

            for i in range(batch_size):
                x_start = centers[i, 0] - self.patch_size // 2
                y_start = centers[i, 1] - self.patch_size // 2
                x_end = x_start + self.patch_size
                y_end = y_start + self.patch_size

                patches[i] = images[i, :3, y_start:y_end, x_start:x_end]
                patches_channel_6[i] = images[i, 4, y_start:y_end, x_start:x_end]

                gc.collect()
                torch.cuda.empty_cache()

            return patches, patches_channel_6, centers



### Autoencoder Class with update loss channel function
class Autoencoder(nn.Module):
    def __init__(self, K=200):
        super(Autoencoder, self).__init__()
        """
        Autoencoder with 8 convolutional and 8 deconvolutional layers, designed for image reconstruction tasks.
        It includes a bottleneck layer with configurable dimensionality and a skip connection to improve gradient flow.

        Architecture:
          - Encoder: A sequence of convolutional layers with batch normalization and Leaky ReLU activations.
          - Bottleneck: A compact representation with `K` channels.
          - Decoder: A mirrored sequence of deconvolutional layers, reconstructing the input image.
          - Skip connection: A shortcut from an intermediate encoder feature (x5) to the decoder to aid training.

        Parameters:
          K (int): Number of bottleneck channels (default: 200).
        """

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128, affine=False)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256, affine=False)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128, affine=False)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64, affine=False)
        self.conv8 = nn.Conv2d(64, K, kernel_size=8, stride=1, padding=0)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(K, 64, kernel_size=8, stride=1, padding=0)
        self.bn8 = nn.BatchNorm2d(64, affine=False)
        self.deconv2 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(128, affine=False)
        self.deconv3 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(256, affine=False)
        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn11 = nn.BatchNorm2d(256, affine=False)
        self.deconv5 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(128, affine=False)
        self.deconv6 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128, affine=False)
        self.deconv7 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(64, affine=False)
        self.deconv8 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
      """
      Performs the forward pass of the autoencoder.

      The input image is first encoded through a series of convolutional layers to obtain a compressed latent representation
      (bottleneck), which is then decoded back to the original image space using transposed convolutions.
      A skip connection is used from an intermediate encoder output (x5) to improve reconstruction quality and training speed.

      Parameters:
        x (Tensor): Input image tensor of shape [B, C, H, W] or [B, H, W, C].

      Returns:
        Tensor: Reconstructed image tensor of shape [B, 3, H, W].
      """
      # Ensure input dimensions are in (batch_size, channels, height, width)
      if x.shape[1] != 3:  # Assume input might be (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)

      # Encoder
      x1 = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
      x2 = F.leaky_relu(self.bn2(self.conv2(x1)), negative_slope=0.2)
      x3 = F.leaky_relu(self.bn3(self.conv3(x2)), negative_slope=0.2)
      x4 = F.leaky_relu(self.bn4(self.conv4(x3)), negative_slope=0.2)
      x5 = F.leaky_relu(self.bn5(self.conv5(x4)), negative_slope=0.2)
      x6 = F.leaky_relu(self.bn6(self.conv6(x5)), negative_slope=0.2)
      x7 = F.leaky_relu(self.bn7(self.conv7(x6)), negative_slope=0.2)
      bottleneck = self.conv8(x7)

      # Decoder
      d1 = F.leaky_relu(self.bn8(self.deconv1(bottleneck)), 0.2)
      d2 = F.leaky_relu(self.bn9(self.deconv2(d1)), 0.2)
      d3 = F.leaky_relu(self.bn10(self.deconv3(d2)), 0.2)
      d3 = torch.cat((d3, x5), dim=1)  # Skip connection
      d4 = F.leaky_relu(self.bn11(self.deconv4(d3)), 0.2)
      d5 = F.leaky_relu(self.bn12(self.deconv5(d4)), 0.2)
      d6 = F.leaky_relu(self.bn13(self.deconv6(d5)), 0.2)
      d7 = F.leaky_relu(self.bn14(self.deconv7(d6)), 0.2)
      reconstructed = self.deconv8(d7)  # No activation here; you can subtract from input

      return reconstructed

    def update_loss_channel(self, indices, patches, reconstructed, images, centers, patch_size=64):
      """
      Computes the pixel-wise reconstruction loss for each patch and writes it to a specific channel
      (fourth index: channel 3) of the corresponding original image.

      This function is typically used in anomaly detection pipelines, where loss maps serve as soft indicators
      of unusual regions. Each patch’s reconstruction loss is written back into the full-sized image at
      the location where the patch was originally sampled from, allowing the system to accumulate
      per-pixel anomaly scores over multiple sampling steps.

      Parameters:
        indices (list of int): Indices mapping each patch to its corresponding image in the batch.
        patches (Tensor): Original image patches of shape [num_patches, 3, patch_size, patch_size].
        reconstructed (Tensor): Reconstructed patches of the same shape.
        images (Tensor): Batch of full images of shape [num_images, 4, H, W] (must include a 4th channel for loss).
        centers (list of tuples): List of (x, y) center coordinates for each patch.
        patch_size (int): Size of the square patch (default: 64).

      Returns:
        Tensor: Patch-wise pixel-level loss maps of shape [num_patches, patch_size, patch_size].
      """

      # Ensure input dimensions are in (batch_size, channels, height, width)
      #if x.shape[1] != 3:  # Assume input might be (batch_size, height, width, channels)
          #x = x.permute(0, 3, 1, 2)
      # Compute the loss map for each patch
      with torch.no_grad():
        loss_map = torch.mean(torch.abs(patches - reconstructed), dim=1)  # Shape: [num_patches, patch_size, patch_size]

        half_patch = patch_size // 2

        # Loop through each patch and update the corresponding image
        for i, index in enumerate(indices):
            center_x, center_y = centers[i]  # Get the center for the current patch

            # Extract patch bounds
            x_start = int(center_x) - half_patch
            y_start = int(center_y) - half_patch
            x_end = x_start + patch_size
            y_end = y_start + patch_size

            # Update the fifth channel of the image at the specified index
            images[index][0][3, y_start:y_end, x_start:x_end] = loss_map[i]



### Predictor Class for Segmentation task with loss profile
class Predictor(nn.Module):
    def __init__(self, input_channels=10):
        super(Predictor, self).__init__()
        """
        Predictor network for binary segmentation, using dilated convolutions to increase receptive field.

        This network takes as input a multi-channel tensor (e.g., loss profiles or feature maps)
        and outputs a single-channel probability mask indicating the likelihood of each pixel
        belonging to the target class (e.g., anomaly region).

        Parameters:
          input_channels (int): Number of input channels. Default is 10, typically corresponding to loss history channels.

        Returns:
          Tensor: A single-channel tensor of probabilities (values between 0 and 1) representing the predicted binary mask.
        """

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=8, dilation=8)
        self.bn4 = nn.BatchNorm2d(4)
        self.output = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
      """
      Forward pass for the predictor.

      Parameters:
        x (Tensor): Input tensor of shape [B, input_channels, H, W], representing multi-channel loss profiles or features.

      Returns:
        Tensor: Output tensor of shape [B, 1, H, W] with values between 0 and 1 after sigmoid activation,
            representing the predicted probability mask for binary segmentation.
      """

      x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
      x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
      x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
      x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
      return torch.sigmoid(self.output(x))

### Device definition for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


### DataSet Class
class ImageLoaderDataset(Dataset):
    def __init__(self, directory_path, transform=None, ignore_folders=None, subfolder=True, include_history=False,
                 autoencoder=None, device='cpu', include_loss_history=True, include_counter=True,
                 patch_size=64, crop_size=128, move_size=24, gray=False):
        """
        A custom PyTorch Dataset for loading images with optional metadata channels
        and integration with autoencoder-based anomaly maps.

        Parameters:
            directory_path (str): Path to the folder containing images.
            transform (callable, optional): A function/transform to apply to the images (e.g., resizing, ToTensor).
            ignore_folders (list, optional): List of subfolders to ignore.
            subfolder (bool): If True, loads images recursively from subfolders.
            include_history (bool): Adds a binary history channel (used to track visited pixels during RL).
            include_loss_history (bool): Adds a layer to track per-pixel anomaly scores (e.g., loss map).
            include_counter (bool): Adds a layer to count how many times a pixel was visited.
            patch_size (int): Size of extracted patches during sampling (used downstream).
            crop_size (int): Size of the initial crop taken from the image.
            move_size (int): Step size for shifting patch center during RL.
            gray (bool): If True, loads images in binary scale (e.g., for ground-truth masks).
            autoencoder (nn.Module, optional): Autoencoder used for computing anomaly maps.
            device (str): Device for autoencoder inference ('cpu' or 'cuda').
        """
        self.directory_path = directory_path
        self.transform = transform if transform else transforms.ToTensor()
        self.ignore_folders = ignore_folders if ignore_folders else []
        self.subfolder = subfolder
        self.include_history = include_history
        self.include_loss_history = include_loss_history
        self.include_counter = include_counter
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.move_size = move_size
        self.gray = gray
        self.group_indices = {}  # Mapping from subfolder to start-end index range
        self.autoencoder = None
        self.device = 'cpu'
        self.image_paths = self._load_image_paths()

    def set_autoencoder(self, model, device='cpu'):
        """
        Sets the autoencoder model used for generating anomaly heatmaps.

        Parameters:
            model (nn.Module): The pre-trained autoencoder model.
            device (str): Device to use for inference ('cpu' or 'cuda').
        """
        self.autoencoder = model
        self.device = device
        self.autoencoder.eval()

    def _load_image_paths(self):
        """
        Loads image paths from the dataset folder (with or without subfolders).

        Returns:
            list: Full paths to image files.
        """
        image_paths = []
        if self.subfolder:
            start_idx = 0
            subfolders = sorted(os.listdir(self.directory_path))
            subfolders = [sf for sf in subfolders if sf not in self.ignore_folders]
            print('Sorted Subfolders:')
            print(subfolders)
            for subfolder in subfolders:
                subfolder_path = os.path.join(self.directory_path, subfolder)
                if os.path.isdir(subfolder_path):
                    image_names = sorted(os.listdir(subfolder_path))
                    for img_name in image_names:
                        print(f'name image in subfolder {subfolder_path}')
                        print(img_name)
                        img_path = os.path.join(subfolder_path, img_name)
                        if img_path.endswith((".png", ".jpg", ".jpeg")):
                            image_paths.append(img_path)
                    end_idx = start_idx + len(image_names) - 1
                    self.group_indices[subfolder] = (start_idx, end_idx)
                    start_idx = end_idx + 1
        else:
            image_names = sorted(os.listdir(self.directory_path))
            for img_name in image_names:
                img_path = os.path.join(self.directory_path, img_name)
                if img_path.endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(img_path)
        return image_paths

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and returns a transformed image with optional metadata channels.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: Image tensor with extra channels (if enabled).
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("1" if self.gray else "RGB")
        img_tensor = self.transform(img)

        # Add empty history layer (binary map tracking visited pixels)
        if self.include_history:
            history_layer = torch.zeros((1, img_tensor.size(1), img_tensor.size(2)))
            img_tensor = torch.cat([img_tensor, history_layer], dim=0)

        # Add empty anomaly score/loss history layer
        if self.include_loss_history:
            loss_history_layer = torch.zeros((1, img_tensor.size(1), img_tensor.size(2)))
            img_tensor = torch.cat([img_tensor, loss_history_layer], dim=0)

        # Add empty counter layer (used for visitation frequency)
        if self.include_counter:
            counter_layer = torch.zeros((1, img_tensor.size(1), img_tensor.size(2)))
            img_tensor = torch.cat([img_tensor, counter_layer], dim=0)

        # If autoencoder is provided, compute fused anomaly map and append it
        if self.autoencoder is not None:
            with torch.no_grad():
                # 1. MAE reconstruction error
                input_to_model = img_tensor[0:3].unsqueeze(0).to(self.device)
                output = self.autoencoder(input_to_model)
                mae_map = torch.mean(torch.abs(output - input_to_model), dim=1, keepdim=True).squeeze(0).cpu()

                # 2. Convert to grayscale for CV2 operations
                gray = TF.rgb_to_grayscale(img_tensor[0:3], num_output_channels=1)
                gray_np = gray.squeeze(0).numpy().astype(np.float32)

                # 3. Local variance map
                kernel_size = 3
                mean = cv2.blur(gray_np, (kernel_size, kernel_size))
                mean_sq = cv2.blur(gray_np ** 2, (kernel_size, kernel_size))
                local_var = mean_sq - mean ** 2
                local_var_tensor = torch.from_numpy(local_var).float()

                # 4. Sobel gradient magnitude
                grad_x = cv2.Sobel(gray_np, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_np, cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = cv2.magnitude(grad_x, grad_y)
                grad_tensor = torch.from_numpy(grad_mag).float()

                # 5. Z-score normalization
                def z_score_norm(tensor):
                    mean = tensor.mean()
                    std = tensor.std()
                    return (tensor - mean) / (std + 1e-8)

                mae_z  = z_score_norm(mae_map)
                var_z  = z_score_norm(local_var_tensor)
                grad_z = z_score_norm(grad_tensor)

                # 6. Weighted fusion of three maps
                fused = (0.7 * mae_z) + (0.1 * var_z) + (0.2 * grad_z)

                # 7. Min-max normalization to [0, 1]
                fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

                # 8. Add fused map as extra channel
                img_tensor = torch.cat([img_tensor, fused.unsqueeze(0)], dim=0)

        return img_tensor


### Load Data
unlabel_train = ImageLoaderDataset(directory_path=unlabel_train_path, transform=transform, subfolder=False)
unlabel_test = ImageLoaderDataset(directory_path=unlabel_test_path, transform=transform, subfolder=False)
label_dataset = ImageLoaderDataset(directory_path=label_path, transform=transform, ignore_folders=['good'],
                                   subfolder=True)
ground_truth = ImageLoaderDataset(directory_path=ground_truth_path, transform=transform, subfolder=True,
                                  include_history=False, include_loss_history=False, include_counter=False, gray=True)

### Add (loss + local variace + sobel filter) channel to images
autoencoder_map = Autoencoder().to('cpu')
autoencoder_map.load_state_dict(torch.load(autoencoder_path, map_location='cpu'))
autoencoder_map.eval()
unlabel_train.set_autoencoder(autoencoder_map, 'cpu')
unlabel_test.set_autoencoder(autoencoder_map, 'cpu')
label_dataset.set_autoencoder(autoencoder_map, 'cpu')

print(unlabel_train[0].shape, unlabel_test[0].shape, label_dataset[0].shape)

random.seed(42)

label_train = []
groundtruth_train = []
label_test = []
groundtruth_test = []

for group, (start_idx, end_idx) in label_dataset.group_indices.items():
    if group != '.ipynb_checkpoints':
      random_indices = random.sample(range(start_idx, end_idx + 1), 5)
      for idx in random_indices:
        label_train.append(label_dataset[idx])
        groundtruth_train.append(ground_truth[idx])

      remaining_indices = [i for i in range(start_idx, end_idx + 1) if i not in random_indices]
      for idx in remaining_indices:
        label_test.append(label_dataset[idx])
        groundtruth_test.append(ground_truth[idx])

default_ground_truth = torch.zeros(1, 256, 256)
default_ground_truth_train = torch.ones(1)
train_data = list(zip(label_train, groundtruth_train)) + [(data, default_ground_truth_train) for data in unlabel_train]
test_data = list(zip(label_test, groundtruth_test)) + [(data, default_ground_truth) for data in unlabel_test]


### Augmentation function
def apply_augmentation(image, label):

    image_rgb = image[:3]
    image_extra = image[3:]

    state = torch.get_rng_state()
    augmented_rgb = augmentation(image_rgb)
    torch.set_rng_state(state)
    augmented_labels = augmentation(label)

    augmented_image = torch.cat([augmented_rgb, image_extra], dim=0)

    del image_rgb, image_extra, augmented_rgb
    gc.collect()
    os.system('echo 1 > /proc/sys/vm/drop_caches')
    return augmented_image, augmented_labels

count_ones = 0
count_not_ones = 0

for _, label in train_data:
    if torch.equal(label, torch.ones(1)):
        count_ones += 1
    else:
        count_not_ones += 1

print(f"Number of samples without label: {count_ones}")
print(f"Number of samples with labels: {count_not_ones}")

final_train = []
num_augmentations = 6

### Augmentation
# Assuming train_data is a list of tuples: (image, ground_truth)
for image, label in train_data:
    final_train.append((image, label))
    if not torch.equal(label, torch.tensor([1.0])):  # Check if labels are not tensor([1])
      # Iterate over each (image, ground_truth) pair in train_data
      for _ in range(num_augmentations):
        augmented_image, augmented_labels = apply_augmentation(image, label)
        #augmented_image = add_gaussian_noise(augmented_image)
        # After augmenting all data in this iteration, append the new augmented data pair
        final_train.append((augmented_image, augmented_labels))

count_ones = 0
count_not_ones = 0

for _, label in final_train:
    if torch.equal(label, torch.ones(1)):
        count_ones += 1
    else:
        count_not_ones += 1

print(f"Number of samples after augmentation without label: {count_ones}")
print(f"Number of samples after augmentation with labels: {count_not_ones}")

vars_to_delete = [
    'ConcatDataset', 'transform', 'augmentation', 'add_gaussian_noise',
    'ImageLoaderDataset', 'apply_augmentation', 'filtering'
    "label1", "unlabel_train_path", "unlabel_test_path", "label_path",
    "ground_truth_path", "transform", "augmentation", "add_gaussian_noise",
    "unlabel_train", "unlabel_test", "image1", "image2", "label_dataset",
    "user_vars", "augmented_image", "augmented_labels", "image", "labels",
    "num_augmentations", "augmented_imgs_list", "augmented_labels_list",
    "train_data", "default_ground_truth", "remaining_indices", "label_train",
    "groundtruth_train", "label_test", "groundtruth_test", "group",
    "start_idx", "end_idxrandom_indices", "idx", "ground_truth",
]

for var in vars_to_delete:
    if var in globals():
        del globals()[var]
        gc.collect()
        os.system('echo 1 > /proc/sys/vm/drop_caches')


train_data = final_train
del final_train
gc.collect()
os.system('echo 1 > /proc/sys/vm/drop_caches')

# Shuffle training data to ensure randomness in sampling
random.shuffle(train_data)
random.shuffle(test_data)

# Initialize the reinforcement learning-based patch sampler
sampler = NeuralBatchSampler(input_size=256, crop_size=128, patch_size=64, move_size=24, action_space=9).to(device)
# Initialize the autoencoder for feature extraction and reconstruction loss estimation
autoencoder = Autoencoder(K=200).to(device)
# Initialize the predictor network for final classification or anomaly scoring
predictor = Predictor(input_channels=10).to(device)


### Custom Entropy loss class for Predictor
class CustomPredictorLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Custom binary segmentation loss with class balancing using a weighted BCE formulation.

        Parameters:
            alpha (float): Weighting factor to balance negative class penalty (default: 0.5).
                           Higher alpha increases penalty for false positives.
        """
        super(CustomPredictorLoss, self).__init__()
        self.alpha = alpha

    def set_alpha(self, new_alpha):
        """
        Update the alpha value for class balancing during training.

        Parameters:
            new_alpha (float): New alpha value to update.
        """
        self.alpha = new_alpha

    def forward(self, y_true, y_pred):
        """
        Compute the forward pass of the custom loss function.

        Parameters:
            y_true (Tensor): Ground truth labels, shape (batch_size, 1, height, width),
                             values should be in {0,1}.
            y_pred (Tensor): Predicted probabilities (after sigmoid), shape (batch_size, 1, height, width),
                             values should be in (0,1).

        Returns:
            loss (Tensor): Scalar loss value.
        """
        # Compute pixel-wise weighted binary cross-entropy loss
        # Positive class is weighted as 1, negative class is weighted as alpha
        pixel_loss = y_true * torch.log(y_pred + 1e-8) + \
                     self.alpha * (1 - y_true) * torch.log(1 - y_pred + 1e-8)

        # Average loss over spatial dimensions (height, width) for each image
        image_loss = torch.mean(pixel_loss, dim=(1, 2, 3))  # Fixed: added missing channel dimension (dim=3)

        # Final loss: mean over batch, and sign flipped to minimize
        loss = -torch.mean(image_loss)

        return loss

### R_cover class
def calculate_R_cover(input):
    """
    Computes the R_cover reward to encourage coverage of diverse regions in the image.

    This reward is designed to penalize the agent for repeatedly selecting the same regions.
    It is computed based on the mean value of the 6th channel of the patch map,
    which typically tracks selection frequency or visitation count per pixel.

    Parameters:
        input (Tensor): A tensor of shape [batch, height, width] representing
                        the 6th channel (coverage frequency map) of the image patches.

    Returns:
        Tensor: A scalar reward value (R_cover) as a PyTorch tensor with gradient tracking.
                Lower mean coverage results in higher reward (due to negative sigmoid).
    """
    if len(input) == 0:
        # Return zero reward if the input is empty, but keep gradient tracking.
        return torch.tensor(0.0, requires_grad=True, device=input.device)

    # Reward is higher when the coverage is low (to encourage exploration)
    return -torch.sigmoid(input.mean())


### R_cover class
def calculate_R_clone(input):
    """
    Computes the R_clone reward, which encourages attention to edges and structural details
    by applying Gaussian blur followed by Sobel edge detection on grayscale patches.

    This reward is high when the extracted patches contain strong edge information,
    incentivizing the agent to sample image regions with high visual complexity.

    Parameters:
        input (Tensor): Tensor of shape [batch, channels, height, width],
                        representing RGB image patches.

    Returns:
        Tensor: A scalar reward value (R_clone) as a PyTorch tensor with gradient tracking.
                Higher edge intensity yields a higher reward.
    """
    if len(input) == 0:
        # Return zero reward for empty input, keeping gradient flow.
        return torch.tensor(0.0, requires_grad=True, device=input.device)

    # Define Sobel filters for detecting horizontal and vertical edges
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=input.device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=input.device).view(1, 1, 3, 3)

    # Convert RGB patches to grayscale using standard luminance weights
    gray_patches = 0.2989 * input[:, 0:1, :, :] + \
                   0.5870 * input[:, 1:2, :, :] + \
                   0.1140 * input[:, 2:3, :, :]

    # Apply a 5x5 Gaussian blur to reduce noise before edge detection
    kernel = torch.tensor([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]], dtype=torch.float32, device=input.device).view(1, 1, 5, 5) / 256.0
    smoothed_patch = F.conv2d(gray_patches, kernel, padding=2)

    # Apply Sobel edge filters to the blurred grayscale patches
    edges_x = F.conv2d(smoothed_patch, sobel_x, padding=1)
    edges_y = F.conv2d(smoothed_patch, sobel_y, padding=1)

    # Compute overall edge intensity using gradient magnitude
    edge_intensity = torch.sqrt(torch.clamp(edges_x ** 2 + edges_y ** 2, min=1e-8)).mean()

    # Return a smooth reward signal using sigmoid
    return torch.sigmoid(edge_intensity)


### Reward loss class ( beta*(R_clone + R_cover) + (1 - beta)*R_pred ): loss_nbs = -RewardLoss
class RewardLoss(nn.Module):
    def __init__(self):
        """
        Custom reward-based loss function combining R_clone, R_cover, and predicted reward (R_pred).
        Encourages both spatial coverage and structural informativeness in selected patches.
        """
        super(RewardLoss, self).__init__()

    def forward(self, patches, patches_c6, R_pred, beta):
        """
        Compute the total reward signal and return its negative as a loss.

        Parameters:
            patches (Tensor): Extracted image patches with shape [batch, channels, height, width].
            patches_c6 (Tensor): Heatmap tensor storing the selection frequency of each pixel,
                                 shape [batch, height, width]. Used for R_cover.
            R_pred (Tensor): Scalar reward value predicted by the predictor network.
            beta (float): Weighting factor ∈ [0, 1] to balance handcrafted and predicted rewards.

        Returns:
            Tensor: A scalar tensor representing the loss (i.e., -R_total).
        """

        # Handcrafted reward encouraging exploration and diversity
        R_cover = calculate_R_cover(patches_c6)

        # Handcrafted reward encouraging edge-rich, information-dense patches
        R_clone = calculate_R_clone(patches)

        # Combine handcrafted rewards and learned reward using beta
        R_total = (beta * (R_clone + R_cover)) + ((1 - beta) * R_pred)

        # Return the negative reward to be minimized as a loss
        return -R_total


# Define the Autoencoder loss (Mean Squared Error) and optimizer (Adam)
autoencoder_loss = nn.MSELoss()  # MSE loss for reconstruction
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)  # Adam optimizer for AE

# Define the Predictor loss (Binary Cross Entropy) and optimizer (Adam)
predictor_loss = CustomPredictorLoss(alpha=1.0)  # BCE loss for classification
predictor_optimizer = optim.Adam(predictor.parameters(), lr=1e-3)  # Adam optimizer for Predictor

# Define the Neural Batch Sampler loss and Optimizer (Adam)
sampler_optimizer = optim.Adam(sampler.parameters(), lr=1e-3)
reward_loss = RewardLoss()

buffer_capacity = 10  # Maximum number of patches stored per image

# Count number of anomalous images in the training data
count = sum(1 for _, ground_truth in train_data
            if isinstance(ground_truth, torch.Tensor) and ground_truth.dim() != 1)

# List of deques, one per anomalous image, storing sampled patches (each with maxlen = buffer_capacity)
h_l = [deque(maxlen=buffer_capacity) for _ in range(count)]

# Ground truth labels corresponding to the anomalous images
y_l = [ground_truth for _, ground_truth in train_data
       if isinstance(ground_truth, torch.Tensor) and ground_truth.dim() != 1]

# Number of normal (non-anomalous) images
countt = len(train_data) - count

# List of deques, one per normal image, storing sampled patches
h_u = [deque(maxlen=buffer_capacity) for _ in range(countt)]

# Buffer used to store tuples (h_l, y_l) for GPU sampling and predictor training
Buffer = []

# Number of patches to extract per training step of the autoencoder.
# Note: actual number used is minibatch_size - 1 (i.e., 32)
minibatch_size = 33

# Tracks the lowest loss observed during predictor training
lowest_loss = np.inf

# Training schedule hyperparameters
K = 20   # Every K (20) autoencoder steps, reinitialize weights
M = 14   # In each 20-step training cycle:
         # - first 10 steps fill the buffer (no predictor/sampler training)
         # - next 4 steps skip sampler training
L = 5000 # Hyperparameter controlling the decay of beta; beta becomes zero after 5000 steps
H = 9000 # Hyperparameter controlling the decay of alpha (class loss weight); alpha becomes zero after 9000 steps

# Initial value for beta (weight between reward and predictor loss)
beta = 1

### Reintialize class with autoencoder(model and optimizer) and h_l, h_u(experience buffer)
def reinitialize(train_data, h_l, h_u, K=200):
    """
    Reinitialize training state, including patch buffers and autoencoder model.

    This function is called to:
    - Free GPU memory
    - Reset the patch memory buffers for anomalous and normal images
    - Reinitialize the autoencoder model with a new bottleneck size

    Parameters:
        train_data: Training dataset (list of tuples: (image, ground_truth))
        h_l: List of deques storing patches for anomalous images
        h_u: List of deques storing patches for normal images
        K:   Bottleneck size for the Autoencoder (number of neurons in the latent layer)

    Returns:
        h_l: Reinitialized deque list for anomalous patches
        h_u: Reinitialized deque list for normal patches
        autoencoder: Reinitialized Autoencoder model
        autoencoder_loss: MSE loss for AE training
        autoencoder_optimizer: Adam optimizer for AE training
    """

    # Move existing patch tensors to CPU and free GPU memory
    h_l = [deque([tensor.detach().cpu() for tensor in dq]) for dq in h_l]
    #h_u = [deque([tensor.detach().cpu() for tensor in dq]) for dq in h_u]  # Optional

    # Delete old patch buffers and clear memory
    del h_l, h_u
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Count anomalous and normal samples
    count = sum(1 for _, ground_truth in train_data
                if isinstance(ground_truth, torch.Tensor) and ground_truth.dim() != 1)
    countt = len(train_data) - count

    # Initialize new deques for patches
    h_l = [deque(maxlen=buffer_capacity) for _ in range(count)]
    h_u = [deque(maxlen=buffer_capacity) for _ in range(countt)]

    # Initialize autoencoder with new bottleneck size
    autoencoder = Autoencoder(K=K).to(device)

    # Define loss function and optimizer
    autoencoder_loss = nn.MSELoss()
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    return h_l, h_u, autoencoder, autoencoder_loss, autoencoder_optimizer

print(f"Memory usage: {psutil.virtual_memory().percent}%")




##################### Train Loops ###############################
j = 1
loss_pred, true_loss_pred = [], []
last_loss_pred, true_last_loss_pred = [], []
acc, f1, auc = [], [], []
last_acc, last_f1, last_auc = [], [], []

# Run for 100 episodes
for episode in range(100):

  # Reset patch-related containers for each episode
  patches, patches_c6, centers, index = [], [], [], []

  # Iterate over training dataset with batch size 1
  train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True)

  for i, (image, _) in enumerate(train_dataloader):
    z = 0
    exist_flag = False

    # Loop until a valid patch is found or selected
    while not exist_flag:
      if z == 0:
        # Initialize the first patch and its center
        patch, channel_6, center = sampler.initial_patches(image)
        center = center[0] # Get center coordinates

        # Store patch and its 6th channel and center patch and index image
        patches.append(patch.squeeze(0).to(device))
        patches_c6.append(channel_6.squeeze(0).squeeze(0).to(device))
        centers.append(center)
        index.append(i)
        del patch, channel_6

        # Free memory every 20 samples
        if j % 20 == 0:
          gc.collect()
          torch.cuda.empty_cache()
          torch.cuda.ipc_collect()
        z+=1
      elif z != 0:
        # Select a product and get its center as input to the neural batch sampler
        crop, center_crop = sampler.select_crops(image, center)

        with torch.no_grad():
          # Reconstruct crop using the autoencoder (only RGB channels) as input to the neural batch sampler
          reconstructed_crop = autoencoder.forward(crop[:, :3, :, :].to(device))
        
        # Prepare input with an additional channel (absolute difference)
        crop = crop.clone().to(device)
        crop[:, 3, :, :] = torch.mean(torch.abs(crop[:, :3, :, :] - reconstructed_crop.detach()), dim=1)

        # Get action probabilities from the sampler
        action_probs = sampler.forward(crop)
        action = torch.multinomial(action_probs, 1)

        # If "stop" action is selected (action == 0), end loop
        if action == 0:
          exist_flag = True
        else:
          # Otherwise, move to new patch location based on action
          patch, channel_6, center = sampler.move_select_patches(image=image, action=int(action[0].item()), center=center_crop)

          # Only add patch if center changed (to avoid duplicates)
          if ((center_crop[0] == center[0]).item()) and ((center_crop[1] == center[1]).item()):
            pass
          else:
            patches.append(patch.squeeze(0).to(device))
            patches_c6.append(channel_6.squeeze(0).to(device))
            centers.append(center)
            index.append(i)

          # Memory cleanup
          del patch, channel_6
          if j % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # More memory cleanup
        del crop, reconstructed_crop, action, action_probs
        if j % 20 == 0:
          gc.collect()
          torch.cuda.empty_cache()
          torch.cuda.ipc_collect()

      # If enough patches are collected, train the autoencoder
      if (len(patches) == minibatch_size) or (len(patches) == minibatch_size - 1):
        # Prepare patch tensors
        patches_c6 = torch.stack(patches_c6).to(device)
        patches_c6_1 = patches_c6.clone().to(device)
        patches = torch.stack(patches).to(device)
        patches_1 = patches.clone().to(device)

        # Autoencoder training step
        autoencoder_optimizer.zero_grad() # remove gradient
        reconstructeds = autoencoder.forward(patches) # Forward pass
        loss_ae = autoencoder_loss(reconstructeds, patches) # Compute reconstruction loss
        loss_ae.backward() # Backpropagation
        autoencoder_optimizer.step() # Update autoencoder weights
        print(f'Epoch: {episode + 1}, Step: {j}, loss AutoEncoder: {loss_ae}')

        # Update loss channel
        autoencoder.update_loss_channel(indices=index, patches=patches, reconstructed=reconstructeds, images=train_data, centers=centers)

        # Cleanup and memory release
        patches = patches.detach().cpu()
        patches_c6 = patches_c6.detach().cpu()
        patches = []
        patches_c6 = []
        centers = []
        index = []
        del reconstructeds, loss_ae
        if j % 20 == 0:
          gc.collect()
          torch.cuda.empty_cache()
          torch.cuda.ipc_collect()


        # Collect loss profiles (unsupervised pixel-level error) from training images
        autoencoder.eval()
        with torch.no_grad():
          l, u = 0, 0
          for ind, (imag, labe) in enumerate(train_data):
            imag = imag.to(device)
            reconstructed = autoencoder.forward(imag.unsqueeze(0)[:, :3, :, :])

            # Only consider labeled data
            if isinstance(labe, torch.Tensor) and labe.dim() != 1:
              h_l[l].append(torch.mean(torch.abs(imag.unsqueeze(0)[:, :3, :, :] - reconstructed), dim=1).squeeze(0).cpu())
              l+=1

            del imag, labe, reconstructed
            torch.cuda.empty_cache()
            if ind % 200 == 0:
              gc.collect()
              torch.cuda.empty_cache()
              torch.cuda.ipc_collect()

        gc.collect()
        os.system('echo 1 > /proc/sys/vm/drop_caches')
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        autoencoder.train()


        # If all class buffers are full, proceed to predictor training
        if all(len(deq) == buffer_capacity for deq in h_l):
          # Extend shared buffer with new (loss_map, label) pairs
          Buffer.extend([(torch.stack([t.to(device) for t in deq]), labels) for deq, labels in zip(h_l, y_l)])

          # Predictor Training Parameters
          num_samples = 80
          num_epochs = 2
          loss_epoch, last_loss_epoch, true_loss_epoch = [], [], []

          # Train predictor on sampled loss profiles
          for epoch in range(num_epochs):
              sampled_indices = torch.randperm(len(Buffer))[:num_samples]
              sampled_data = [Buffer[i] for i in sampled_indices]
              loss_profiles_sample = [sampled_data[i][0] for i in range(num_samples)]
              labels_samples = [sampled_data[i][1] for i in range(num_samples)]

              # Detach and move to CPU to release GPU memory
              sampled_data = [(tensor1.to('cpu'), tensor2.to('cpu')) for tensor1, tensor2 in sampled_data]
              sampled_indices = [tensor.detach().cpu() for tensor in sampled_indices]
              del sampled_data, sampled_indices
              if j % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

              loss_profiles_sample = torch.stack(loss_profiles_sample).to(device)
              labels_samples = torch.stack(labels_samples).to(device)

              predictor_optimizer.zero_grad() # Clear previous gradients

              prediction = predictor(loss_profiles_sample) # Forward pass through the predicton

              loss_p = predictor_loss(labels_samples.squeeze(), prediction.squeeze())  # use the correct loss function

              loss_p.backward() # Backpropagation
              predictor_optimizer.step() # Update model parameters

              ### True Loss (based on BCE)
              with torch.no_grad():
                y_pred_sigmoid = prediction.squeeze()
                pixel_loss = labels_samples.squeeze() * torch.log(y_pred_sigmoid + 1e-8) + (1 - labels_samples.squeeze()) * torch.log(1 - y_pred_sigmoid + 1e-8)
                image_losss = torch.mean(pixel_loss, dim=(1, 2))
                losss = -torch.mean(image_losss)
                true_loss_epoch.append(losss.item())
              ###

              # Cleanup
              del prediction, y_pred_sigmoid, pixel_loss, image_losss, losss
              if j % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
              print(f'Epoch: {episode + 1}, Step: {j}, loss Predicton: {loss_p};           Epoch for Predictor:{epoch + 1}')
              loss_epoch.append(loss_p.item())

          # Store average loss for logging
          loss_pred.append(sum(loss_epoch) / len(loss_epoch))
          true_loss_pred.append(sum(true_loss_epoch) / len(true_loss_epoch))

          # Store average loss for last step cycle
          if ((j % K) == 0) and (j != 0):
            last_loss_pred.append(sum(loss_epoch) / len(loss_epoch))
            true_last_loss_pred.append(sum(true_loss_epoch) / len(true_loss_epoch))

            
          del loss_epoch
          del Buffer
          Buffer = []

          # Evaluation metrics on predictor
          with torch.no_grad():
            predictor.eval()
            probs = predictor(loss_profiles_sample).squeeze().cpu()
            binary_preds = (probs >= 0.5).float()
            true_labels = labels_samples.detach().cpu().numpy().flatten()
            y_pred = binary_preds.detach().cpu().numpy().flatten()

            y_score = binary_preds.view(-1).detach().cpu().numpy()
            y_prob = probs.view(-1).detach().cpu().numpy()

            f1_v = f1_score(true_labels, y_pred)
            auc_v = roc_auc_score(true_labels, y_prob)
            acc_v = accuracy_score(true_labels, y_score)

            acc.append(acc_v)
            f1.append(f1_v)
            auc.append(auc_v)
            if ((j % K) == 0) and (j != 0):
              last_acc.append(acc_v)
              last_f1.append(f1_v)
              last_auc.append(auc_v)
            print(f"Metrics for Prediction Model with Val_Data---   Accuracy: {acc_v}, F1 Score: {f1_v}, AUC: {auc_v}")

            # Cleanup
            del acc_v, f1_v, auc_v, probs, true_labels, binary_preds
            if j % 20 == 0:
              gc.collect()
              os.system('echo 1 > /proc/sys/vm/drop_caches')
              torch.cuda.empty_cache()
              torch.cuda.ipc_collect()
          predictor.train()

          if j % 10 == 0:
            gc.collect()
            os.system('echo 1 > /proc/sys/vm/drop_caches')
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
          Buffer = []

        # Update sampler (NBS) if condition met
        if ((j % K) > M) and (all(len(deq) == buffer_capacity for deq in h_l)):

          sampler_optimizer.zero_grad()

          # Detach to prevent gradients flowing back to Predictor
          detached_loss_profiles_sample = loss_profiles_sample.detach().to(device)
          detached_prediction = predictor(detached_loss_profiles_sample)
          detached_loss_p = predictor_loss(labels_samples.squeeze(), detached_prediction.squeeze())

          # Calculate reward for NBS using negative predictor loss
          R_pred = - detached_loss_p

          # Compute sampler loss based on reward
          loss_nbs = reward_loss(patches_1, patches_c6_1, R_pred, beta)

          # Backpropagation and optimizer step
          loss_nbs.backward()
          sampler_optimizer.step()

          print(f'Epoch: {episode + 1}, Step: {j}, loss NBS: {loss_nbs}')

          # Clean up and free memory
          del R_pred, loss_nbs
          del detached_loss_profiles_sample, detached_prediction, patches_c6_1, patches_1, loss_profiles_sample, labels_samples
          if j % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Save models and reinitialize every K steps
        if ((j % K) == 0) and (j != 0):
          torch.save(predictor.state_dict(), f'predictor_step_{j}.pth')
          torch.save(sampler.state_dict(), f'sampler_step_{j}.pth')
          print(f'Models saved at step {j}')

          # Reinitialize AutoEncoder and loss profiles memory banks
          h_l, h_u, autoencoder, autoencoder_loss, autoencoder_optimizer = reinitialize(train_data, h_l, h_u)
          print(f'j: {j};   ReInitial AutoEncoder & h_l, h_u')


        # Update beta and alpha (for reward and loss predictor weighting)
        beta = max(0.0, 1 - (j / L))
        print(f'Update Beta: {beta}, Step: {j}')
        new_alpha = max(1 - (j / H), 0.0)
        predictor_loss.set_alpha(new_alpha)
        print(f"Alpha updated to {new_alpha},  Step {j}")

        j += 1 # Increase step counter

        # Save plots and metrics every 150 steps
        if j % 150 == 0:
          fig, axs = plt.subplots(2, 2, figsize=(14, 10))

          # Plot predictor loss
          axs[0, 0].plot(loss_pred, label='Loss', color='red')
          axs[0, 0].plot(true_loss_pred, label='True Loss', color='blue')
          axs[0, 0].set_title('Predictor Loss')
          axs[0, 0].set_xlabel('Step')
          axs[0, 0].set_ylabel('Average Loss')
          axs[0, 0].legend()
          axs[0, 0].grid(True)
 
          # Plot metrics: Accuracy, F1 Score, AUC
          axs[0, 1].plot(acc, label='Accuracy', color='blue')
          axs[0, 1].plot(f1, label='F1 Score', color='green')
          axs[0, 1].plot(auc, label='AUC', color='purple')
          axs[0, 1].set_title('Evaluation Metrics')
          axs[0, 1].set_xlabel('Step')
          axs[0, 1].set_ylabel('Score')
          axs[0, 1].legend()
          axs[0, 1].grid(True)
 
          # Plot recent predictor loss (last step cycles)
          axs[1, 0].plot(last_loss_pred, label='Loss', color='red')
          axs[1, 0].plot(true_last_loss_pred, label='True Loss', color='blue')
          axs[1, 0].set_title('last cycles Predictor Loss')
          axs[1, 0].set_xlabel('Step')
          axs[1, 0].set_ylabel('Average Loss')
          axs[1, 0].legend()
          axs[1, 0].grid(True)

          # Plot recent metrics (last step cycles)
          axs[1, 1].plot(last_acc, label='Accuracy', color='blue')
          axs[1, 1].plot(last_f1, label='F1 Score', color='green')
          axs[1, 1].plot(last_auc, label='AUC', color='purple')
          axs[1, 1].set_title('Evaluation Metrics for last cycles')
          axs[1, 1].set_xlabel('Step')
          axs[1, 1].set_ylabel('Score')
          axs[1, 1].legend()
          axs[1, 1].grid(True)

          plt.tight_layout()
          plt.show()
          plt.close(fig)

        # Final cleanup and memory release
        gc.collect()
        os.system('echo 1 > /proc/sys/vm/drop_caches')
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()