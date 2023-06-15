import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class FVUSMDataset(Dataset):
    """ FVUSMDataset for FV-USM-processed
        root: Root dir for the FVUSM dataset.
        transform: Trsansforms for images.
        sample_per_class: How many samples in each class.
        mode: Train set or test set.
        inter_aug: Inter augmentation, 'LF' or 'TB'

    """

    def __init__(self, root, transforms, sample_per_class, mode='train', inter_aug=''):
        self.transform = transforms
        self.files = sorted(glob.glob(os.path.join(root) + '/*.*'))
        self.class_num = len(self.files) // sample_per_class // 2  # why divide 2?
        self.labels = np.arange(self.class_num).repeat(sample_per_class)  # generate labels
        self.img_data = []
        self.mode = mode
        if mode == 'train':
            # trainset from 0 - 2951
            for i in np.arange(0, len(self.files) // 2):
                with open(os.path.join(root, self.files[i]), 'rb') as f:
                    img = Image.open(f)
                    self.img_data.append(img.copy())
                    img.close()
            # self.img_data = [Image.open(os.path.join(root, self.files[i]))
            # for i in np.arange(0, len(self.files) // 2)]

        elif mode == 'test':
            # trainset from 2952 - 5903
            for i in np.arange(len(self.files) // 2, len(self.files)):
                with open(os.path.join(root, self.files[i]), 'rb') as f:
                    img = Image.open(f)
                    self.img_data.append(img.copy())
                    img.close()
            # self.img_data = [Image.open(os.path.join(root, self.files[i]))
            # for i in np.arange(len(self.files) // 2, len(self.files))]

        if inter_aug == 'LF':  # left-right flip
            self.img_data.extend([
                self.img_data[i].transpose(Image.FLIP_LEFT_RIGHT)
                for i in np.arange(0, len(self.img_data))
            ])
            aug_classes = np.arange(self.class_num, self.class_num * 2).repeat(sample_per_class)
            self.labels = np.concatenate([self.labels, aug_classes])
            self.class_num = self.class_num * 2
        elif inter_aug == 'TB':  # top-bottom flip
            self.img_data.extend([
                self.img_data[i].transpose(Image.FLIP_TOP_BOTTOM)
                for i in np.arange(0, len(self.img_data))
            ])
            aug_classes = np.arange(self.class_num, self.class_num * 2).repeat(sample_per_class)
            self.labels = np.concatenate([self.labels, aug_classes])
            self.class_num = self.class_num * 2

    def __getitem__(self, index):
        image = self.img_data[index]
        image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.img_data)


class BalancedBatchSampler(torch.utils.data.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples
    n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples, drop_last=False):
        self.labels = np.array(dataset.labels)
        self.drop_last = drop_last

        self.labels_set = list(set(self.labels))
        self.label_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in self.labels_set
        }
        for label in self.labels_set:
            np.random.shuffle(self.label_to_indices[label])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    np.random.choice(self.label_to_indices[class_], self.n_samples, replace=False))
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
