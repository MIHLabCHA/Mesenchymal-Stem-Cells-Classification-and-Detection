# Load libraries
import torch
import os
import glob
import shutil
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision import datasets
from utils.aux_funcs import create_folder
from retinanet.dataloader import Normalizer, Resizer, collater


# Data preprocessing
def data_transformation(imgsz, mu, std):
    data_T = T.Compose([
        T.Resize(size=(imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize(mu, std)
    ])
    return data_T


# Create a custom dataset for test images
def create_test_loader(input_folder, imgsz, mu, std, batch_size=8):
    # create a subfolder to comply with datasets.ImageFolder
    create_folder(os.path.join(input_folder, '0'))
    
    # grab images
    image_types = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    files_grabbed = []
    for files in image_types:
        files_grabbed.extend(glob.iglob(os.path.join(input_folder, files)))
    
    # throw error if no images detected
    assert len(files_grabbed)>0, 'supported image extensions: .jpg, .jpeg, .png, .bmp, .tif, .tiff'

    # move images to the subfolder
    for file in files_grabbed:
        shutil.move(file, os.path.join(input_folder, '0'))
    
    # create dataset and dataloader
    test_dataset = datasets.ImageFolder(input_folder, transform=data_transformation(imgsz=imgsz, mu=mu, std=std))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1 if len(files_grabbed)==1 else batch_size)

    return test_loader


# Re-organize image folder after inference
def folder_reorganize(input_folder):
    image_types = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    if os.path.exists(os.path.join(input_folder, '0')): # if folder exists
        if os.listdir(os.path.join(input_folder, '0')): # if folder not empty
            files_grabbed = []
            for files in image_types: # grab images
                files_grabbed.extend(glob.iglob(os.path.join(input_folder, '0', files)))
            for file in files_grabbed: # move images back to input_folder
                shutil.move(file, input_folder)
        shutil.rmtree(os.path.join(input_folder, '0')) # remove folder


# Dataset for detector
class CCD(Dataset):
    def __init__(self, input_folder, transforms=None):
        super().__init__()
        self.input_folder = input_folder
        self.transforms = transforms
        image_types = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        files_grabbed = []
        for files in image_types:
            files_grabbed.extend(glob.iglob(os.path.join(input_folder, files)))
        self.files_grabbed = files_grabbed
    
    def __getitem__(self, index: int):
        image = cv2.imread(self.files_grabbed[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.
        boxes = np.zeros((1, 5)) # just for complying with transform functions
        sample = {'img': image, 'annot': boxes}

        if self.transforms:
            sample = self.transforms(sample)
        
        return sample
    
    def __len__(self)->int:
        return len(self.files_grabbed)


# Create a custom dataset for detection model
def create_detection_test_loader(input_folder):
    test_dataset = CCD(input_folder, T.Compose([Normalizer(), Resizer()]))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collater)
    return test_loader