# Load libraries
import numpy as np
import os
import torch
from torch import nn
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer
from utils import aux_funcs, model, train, cc_dataset


# Global settings
RANDOM_STATE = 10
data_source = 'A_singlestack'
data_path = "datasets/"+data_source+"/training_data/image_classification/"
TRAIN = "train_all"
TEST = "test"
imgsz = 300
mu = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)


# Prepare training dataset
train_dataset = datasets.ImageFolder(os.path.join(data_path, TRAIN), transform=cc_dataset.data_transformation(imgsz=imgsz, phase='train', mu=mu, std=std))
train_dataset_noaug = datasets.ImageFolder(os.path.join(data_path, TRAIN), transform=cc_dataset.data_transformation(imgsz=imgsz, phase='test', mu=mu, std=std))
test_dataset = datasets.ImageFolder(os.path.join(data_path, TEST), transform=cc_dataset.data_transformation(imgsz=imgsz, phase='test', mu=mu, std=std))

label_binarizer = LabelBinarizer().fit(train_dataset.targets)

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_dataset.targets), y=train_dataset.targets)
class_weights = torch.Tensor(class_weights)


# Train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
current_path = os.getcwd()
save_path = os.path.join(current_path, 'export')

test_model = model.CCCL(FE_arch='ResNet50', pretrained=False, channels=3, num_classes=4)
outputs = train.kfold_train(with_NNI=False, 
                            logit_calibration=False, 
                            device=device, 
                            model=test_model, 
                            train_dataset=train_dataset, 
                            train_dataset_noaug=train_dataset_noaug, 
                            test_dataset=None, 
                            label_binarizer=label_binarizer, 
                            criterion=nn.CrossEntropyLoss(weight=class_weights.to(device)), 
                            folds=5, 
                            max_epochs=200, 
                            batch_size=8, 
                            learning_rate=1.5e-06, 
                            weight_decay=1e-5, 
                            early_stop_threshold=20, 
                            early_stop_criterion='auc_score', 
                            save_path=save_path, 
                            TTA=False, 
                            random_state=RANDOM_STATE)
