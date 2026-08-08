import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class SSDDataset(Dataset):
    """
    A custom torch dataset dedicated to the SSD input format that overwrittes 
    methods to return Dataloader's objects.
    """
    def __init__(self, datapoints_list:list, labels_list:list, 
                 data_transform:transforms.Compose=None, target_transform:transforms.Compose=None):
                
        self.datapoints_list = datapoints_list
        self.labels_list = labels_list

        self.data_transform = data_transform
        self.target_transform = target_transform

        # The targets remain loaded, so they are augmented only once
        # In the case of segmentation for instance, the mask (which is an image) would likely be 
        # loaded during the __getitem__ process due to memory limitations
        if self.target_transform:
            self.labels_list = [self.target_transform(target) for target in self._preprocess_labels()]
        else:
            self.labels_list = [torch.Tensor(target) for target in self._preprocess_labels()]

    def __len__(self):
        return len(self.datapoints_list)
    
    def _preprocess_labels(self):
        """
        Reads the content of the label files in self.labels_list, and loads it only once at startup into the system RAM.
        """
        labels_list = [0] * len(self.labels_list)
        for idx, label_file_path in enumerate(self.labels_list):
            
            with open(label_file_path, 'r') as file:
                labels = []
                for line in file:
                    parts = line.strip().split(',')
                    class_id, x_min, y_min, x_max, y_max = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    labels.append( [class_id, x_min, y_min, x_max, y_max])
                labels_list[idx] = np.array(labels)

        self.labels_list = labels_list

        return self.labels_list

    def __getitem__(self, idx):
        """
        Loads the datapoint and its labels as arrays, and returns the tensors expected by the model
        """
        
        image = Image.open(os.path.normpath(self.datapoints_list[idx])).convert('RGB')


        # The datapoints are loaded per batch, so they are augmented every epoch
        if self.data_transform:
            image = self.data_transform(image).float() / 255
        else:
            image = torch.Tensor(image).float() / 255

        target = self.labels_list[idx] # (class, x_min, y_min, x_max, y_max, class_txt)
        boxes = target[0, :, -5:-1].float()
        labels = target[0, :, -1].long()

        return image, boxes, labels

    def _collate_fn(self, batch):
        """
        Flattens datapoint/multilabels pairs along the batch axis.

        Since an image may have multiple targets, this functions takes a datapoint/multilabels pair, and returns 
        a tensor of duplicated images, where each image is paired with a single label row from the multilabel tensor.

        Example 
        -------
        Input Batch:
        >>> image = torch.Tensor((n, n))
        >>> labels = torch.Tensor((m, 4))

        Output :
        >>> image = torch.Tensor((m, n, n))
        >>> labels = torch.Tensor((m, 1, 4))
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels
