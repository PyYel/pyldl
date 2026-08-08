import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import cv2
import json
import abc
from abc import abstractmethod, ABC


class Sampler(ABC):
    """
    The sampler handles to data gathering request made to the database. It follows the strategies
    specified during the Active Learning loop.
    """

    def __init__(self) -> None:
        """
        This base sampler handles all the pipelines step without using any data selection 
        strategy, but keeping every datapoint that is already labellized.

        """    


    def split_in_two(self, test_size: float = 0.25, datapoints_list: list[str] = None, labels_list: str = None):
        """
        Splits the querried data into a training and testing batch. Can also be used out of the sampling
        pipeline as a util by overwriting the ``<datapoints_list>`` and/or ``<labels_list>`` inputs.

        Args
        ----
        - test_size: the percentage of batch data to allocate to the testing loop. Thus it won't be used during 
        the whole training process.
        - datapoints_list: the list of paths as described in the ``<from_DB>`` method 
        - labels_list: the list of label tuples as described in the ``<from_DB>`` method 
        """
        if datapoints_list:
            self.datapoints_list = datapoints_list
        if labels_list:
            self.labels_list = labels_list

        if test_size < 1 and test_size > 0:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.datapoints_list, self.labels_list, test_size=test_size)
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = [], self.datapoints_list, [], self.labels_list
        return self.X_train, self.X_test, self.Y_train, self.Y_test


    def send_to_dataloader(self, 
                           dataset:Dataset, 
                           data_transform=None, 
                           target_transform=None,
                           chunks: int = 1, 
                           batch_size: int = None, 
                           num_workers: int = 0, 
                           drop_last: bool = True,
                           shuffle: bool = True):
        """
        Returns a training and testing dataloaders objects from the sampled ``datapoints_list`` and ``labels_list``.

        Args
        ----
        - dataset: a torch Dataset subclass that is compatible with the performed task (a forciori the loaded model)
        - transform: a short datapoints preprocessing pipeline, that should be model specific 
        (such as resizing an image input, or vectorizing a word...) and data specific (normalizing...)
        - chunks: the number of batch to divide the dataset into
        """

        # Custom datasets
        self.train_dataset = dataset(datapoints_list=self.X_train, labels_list=self.Y_train, 
                                     data_transform=data_transform, target_transform=target_transform)
        self.test_dataset = dataset(datapoints_list=self.X_test, labels_list=self.Y_test,
                                     data_transform=data_transform, target_transform=target_transform)
        
        # The batch_size parameter has priority over the number of chunks 
        if chunks and not batch_size:
            train_batch_size = self.train_dataset.__len__()//chunks
            test_batch_size = self.test_dataset.__len__()//chunks
        else:
            train_batch_size = batch_size
            test_batch_size = batch_size

        # Dataloader required for the training loop
        if self.train_dataset: 
            self.train_dataloader = DataLoader(self.train_dataset, 
                                            batch_size=train_batch_size, 
                                            shuffle=shuffle, drop_last=drop_last, 
                                            collate_fn=self.train_dataset._collate_fn,
                                            num_workers=num_workers)
        else: # Training is empty to avoid errors
            self.train_dataloader = []
        # Dataloader required for the testing loop
        self.test_dataloader = DataLoader(self.test_dataset, 
                                          batch_size=test_batch_size,
                                          shuffle=shuffle, drop_last=False, 
                                          collate_fn=self.train_dataset._collate_fn,
                                          num_workers=num_workers)

        return self.train_dataloader, self.test_dataloader


    def _get_classes(self, batch_dict: dict, task_name: str):
        """
        Returns the list of unique classes, i.e. the labelling options
        """

        if task_name not in batch_dict["data"].keys():
            print(f"Sampler >> Batch '{batch_dict['metadata']['batch_name']}' is not created to handle {task_name} task")
            return False

    def _delete_file(self, file_path: str):
        """
        Deletes a local file
        """
        try:
            os.remove(file_path)
        except:
            None
        return None

    def _check_batch_task_compatibility(self, batch_dict: dict, task_name: str):
        """
        Checks if a batch is compatible with the chosen labelling task
        """

        if task_name not in batch_dict["data"].keys():
            print(f"Sampler >> Batch '{batch_dict['metadata']['batch_name']}' is not created to handle {task_name} task")
            return False

        return True
