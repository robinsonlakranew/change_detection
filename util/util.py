"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os, math, random
import ntpath
from collections import namedtuple

def save_images(images, img_dir, name):
    """save images in img_dir, with name
    iamges: torch.float, B*C*H*W
    img_dir: str
    name: list [str]
    """
    for i, image in enumerate(images):
        print(image.shape)
        image_numpy = tensor2im(image.unsqueeze(0),normalize=False)*255
        basename = os.path.basename(name[i])
        print('name:', basename)
        save_path = os.path.join(img_dir,basename)
        save_image(image_numpy,save_path)


def save_visuals(visuals,img_dir,name):
    """
    """
    name = ntpath.basename(name)
    name = name.split(".")[0]
    print(name)
    # save images to the disk
    for label, image in visuals.items():
        image_numpy = tensor2im(image)
        img_path = os.path.join(img_dir, '%s_%s.png' % (name, label))
        save_image(image_numpy, img_path)


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if normalize:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def define_parameters():
    # Dictionary to hold parameters

    parameters = {
        'dataroot': './samples', # Path to the dataset directory
        'name': 'pam', # Name of the experiment or model
        'gpu_ids': '-1', # GPU IDs to use, '-1' means using CPU
        'checkpoints_dir': './checkpoints', # Directory to save model checkpoints
        'model': 'CDFA', # Model name or type
        'input_nc': 3, # Number of input channels (e.g., 3 for RGB images)
        'output_nc': 3, # Number of output channels
        'arch': 'mynet3', # Architecture type or name
        'f_c': 64, # Number of filters in the convolution layers
        'n_class': 2, # Number of output classes
        'init_type': 'normal', # Initialization method for model weights
        'init_gain': 0.02, # Gain for the initialization method
        'SA_mode': 'PAM', # Mode for Self-Attention, 'PAM' in this case
        'ds': 1, # Downsampling factor
        'angle': 0, # Angle parameter, for data augmentation
        'istest': False, # Flag to indicate if the mode is testing
        'serial_batches': False, # If True, loads data in a fixed order
        'num_threads': 4, # Number of threads for data loading
        'batch_size': 1,  # Number of samples per batch
        'load_size': 286, # Size to resize images before cropping
        'crop_size': 256, # Size of the crop applied to images
        'max_dataset_size': float("inf"), # Maximum size of the dataset to load
        'preprocess': 'none1',  # Preprocessing method to apply to images
        'no_flip': True,  # Disable horizontal flipping during preprocessing
        'no_flip2': True, # Possibly another flag related to flipping
        'display_winsize': 256, # Display window size
        'epoch': 'pam', # Specific epoch to load or use
        'load_iter': 0, # Iteration to load the model from
        'verbose': False, # Verbosity flag for logging
        'suffix': '', # Suffix for saving the model or results
        'num_threads': 0,  # Number of threads for data loading
        'batch_size': 1,  # Batch size for testing
        'serial_batches': True,  # Load images in order
        'no_flip': True,  # Disable flipping
        'no_flip2': True, # Disable flipping
        'display_id': -1,  # ID for displaying results; -1 means no display
        'phase': 'test', # Phase of the experiment, 'test' in this case
        'preprocess': 'none1',  # Preprocessing method
        'isTrain': False,  # Indicates that this is not training mode
        'aspect_ratio': 1,  # Aspect ratio for resizing images
        'eval': True, # Evaluation mode flag
        'results_dir': './samples/output/', # Directory to save the results
        'data_dir': './samples', # Directory of the dataset
        'num_test': np.inf # Number of test samples to use
    }

    # Convert the dictionary into a named tuple for easier access
    Opt = namedtuple('Opt', parameters.keys())
    opt = Opt(**parameters)

    return opt

def get_params(opt, size, test=False):
  """
  This function generates random parameters for image transformations based on the parameters and image size.

  Args:
    opt: A dictionary containing options for image preprocessing.
    size: A tuple representing the original image size (width, height).
    test: A boolean flag indicating whether it's training or testing phase (default: False).

  Returns:
    A dictionary containing transformation parameters:
      - crop_pos: (x, y) coordinates for random cropping (if applicable).
      - flip: A boolean indicating horizontal flip (True) or not (False).
      - angle: A random rotation angle in degrees (if applicable).
  """

  w, h = size  # Unpack width and height from the size tuple
  new_h = h  # Initialize new height (potentially modified)
  new_w = w  # Initialize new width (potentially modified)
  angle = 0  # Initialize rotation angle (default: 0 degrees)

  # Apply resize and crop transformation based on options
  if opt.preprocess == 'resize_and_crop':
    new_h = new_w = opt.load_size  # Resize both height and width to the specified size

  # Apply random rotation during training
  if 'rotate' in opt.preprocess and test is False:
    angle = random.uniform(0, opt.angle)  # Generate random angle within the specified range
    # Calculate new width and height considering rotation
    new_w = int(new_w * math.cos(angle * math.pi / 180) + new_h * math.sin(angle * math.pi / 180))
    new_h = int(new_h * math.cos(angle * math.pi / 180) + new_w * math.sin(angle * math.pi / 180))
    # Ensure new width and height are within image bounds
    new_w = min(new_w, new_h)
    new_h = min(new_w, new_h)

  # Calculate random crop coordinates if necessary
  x = random.randint(0, np.maximum(0, new_w - opt.crop_size))  # x-coordinate within valid range
  y = random.randint(0, np.maximum(0, new_h - opt.crop_size))  # y-coordinate within valid range

  # Apply random horizontal flip
  flip = random.random() > 0.5  # Random boolean for flip (True or False)

  # Return a dictionary containing the generated parameters
  return {'crop_pos': (x, y), 'flip': flip, 'angle': angle}

