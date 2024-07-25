import numpy as np
import torch
import torchvision
import os
import matplotlib.patches as patches
from bs4 import BeautifulSoup
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


# Classification
def data_transformation(imgsz=512, phase=None, mu=None, std=None):
    if phase=='train':
        data_T = T.Compose([
            T.Resize(size=(imgsz, imgsz)),
            T.RandomRotation(degrees=(-20, +20)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(),
            T.GaussianBlur(),
            T.RandomAdjustSharpness(),
            T.RandomAutocontrast(),
            T.ToTensor(),
            T.Normalize(mu, std)
        ])
    elif phase=='test': # no augmentation
        data_T = T.Compose([
            T.Resize(size=(imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize(mu, std)
        ])
    return data_T


# Synthesis
def data_transformation_synth(imgsz=300, phase=None):
    """
    ToTensor() and Normalize() are performed within the __getitem__ of the custom dataset
    """
    if phase=='train':
        data_T = T.Compose([
            T.Resize(size=(imgsz, imgsz)),
            T.ToTensor()
        ])
    elif phase=='test':
        data_T = T.Compose([
            T.Resize(size=(imgsz, imgsz)),
            T.ToTensor()
        ])
    return data_T


# Detection
def generate_box(obj):
    xmin = float(obj.find('xmin').text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    if obj.find('name').text == "Flatten":
        return 0
    return 1

def generate_target(file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all("object")

        num_objs = len(objects)

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))

        boxes = torch.as_tensor(boxes, dtype=torch.float32) 
        labels = torch.as_tensor(labels, dtype=torch.int64) 
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        return target

def plot_image_from_output(img, annotation):
    img = img.cpu().permute(1,2,0)
    rects = []

    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 0 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        else:
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')

        rects.append(rect)

    return img, rects

class MaskDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + 'xml'
        img_path = os.path.join(self.path, file_image)
        
        if 'val' in self.path:
            label_path = os.path.join(self.path.replace('val_images', 'val_annotations'), file_label)
        else:
            label_path = os.path.join(self.path.replace('train_images', 'train_annotations'), file_label)

        img = Image.open(img_path).convert("RGB")
        target = generate_target(label_path)
        
        to_tensor = torchvision.transforms.ToTensor()

        if self.transform:
            img, transform_target = self.transform(np.array(img), np.array(target['boxes']))
            target['boxes'] = torch.as_tensor(transform_target)

        # change to tensor
        img = to_tensor(img)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))