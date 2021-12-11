"""
Classes to load the correlation data, create in train/test splits, and apply preprocessing.
"""

from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
import torch

import pandas as pd
import numpy as np
import os

class TrainTestSplitter(object):
    # Used to obtain consistent train and test data indices
    # across Datasets
    def __init__(self, test_ratio=.2, seed=42):
        self.test_ratio=.2
        self.seed = 42
    
    def sample_indices(self, n, train=True):
        # Returns samples of train or test set, with full data size n
        index_cutoff = int(n * self.test_ratio)
        # Get consistent permutation of data indices
        rng = np.random.default_rng(42)
        per = rng.permutation(n)
        # Select train or test set
        if train:
            return per[index_cutoff:]
        else:
            return per[:index_cutoff]


class CorrelationDataset(Dataset):
    # Dataset for the correlation data
    def __init__(self, responses_file, img_dir, train_test_sampler, train=True, transform=None):
        # Select train or test samples from responses file
        df_data = pd.read_csv(responses_file, dtype={'id': 'O', 'corr': np.float32})
        indices = train_test_sampler.sample_indices(len(df_data), train)
        self.data = df_data.iloc[indices]
        # Store image directory and transformation pipeline
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image and look up label
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0] + ".png")
        image = pil_loader(img_path)
        label = self.data.iloc[idx, 1]
        # Transform image and label
        if self.transform:
            image, label = self.transform(image, label)
        return image, label

class CorrelationTransform(object):
    # Crops image to only include the graph, resizes it and probabilistically
    # flips it if it is training data.
    def __init__(self, train, size=32, p_flip=.5):
        self.train = train
        self.size = size
        self.p_flip = p_flip

    def __call__(self, image, label):
        # Crop and resize
        image = transforms.functional.resized_crop(
            image,
            2, 21, 127, 127, (self.size, self.size),
            interpolation=True
        )
        # Probabilistically flipping train data
        if self.train:
            # Horizontal flip
            if np.random.rand() < self.p_flip:
                image = transforms.functional.hflip(image)
                label *= -1
            # Vertical flip
            if np.random.rand() < self.p_flip:
                image = transforms.functional.vflip(image)
                label *= -1
        # Make tensor from PIL image
        image = transforms.ToTensor()(image)
        label = np.float32(label)

        return image, label

