#To import configuration parameters
import config
# To load the datasets from .pickle files
import hickle
import os
# Needed to randomize indices
import random
# Needed to split the dataset into train/test subsets
from torch.utils.data import Dataset, Subset, ConcatDataset
# Our own files
from create_dataset import ScanDataSet
#Standard libraries
import numpy as np


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def split_dataset_one_random_sample_from_each_class(original_dataset, augmented_dataset):
    # Import the configuration parameter nrOfAugmentations
    nrOfAugmentations = config.NumberOfAugmentations

    # Extract the train/test indices from the original dataset
    all_indices = np.linspace(0, len(original_dataset)-1, len(original_dataset)).astype(int)
    test_indices = np.asarray(random.sample(range(0, len(original_dataset)-1), nrOfAugmentations))
    original_test_indices = test_indices
    original_train_indices = np.delete(all_indices, original_test_indices)

    # Extract the train/test indices from the augmented dataset
    augmented_test_indices = original_test_indices
    for i in range(1, nrOfAugmentations):
        augmented_test_indices = np.append(augmented_test_indices, test_indices + i*len(original_dataset))

    all_indices_augmented = np.linspace(0, len(augmented_dataset)-1, len(augmented_dataset)).astype(int)
    augmented_train_indices = np.delete(all_indices_augmented, augmented_test_indices)

    # Split the datasets
    original_train_dataset, original_test_dataset = [Subset(dataset = original_dataset, indices = original_train_indices),Subset(dataset = original_dataset, indices = original_test_indices)]
    augmented_train_dataset, augmented_test_dataset = [Subset(dataset = augmented_dataset, indices = augmented_train_indices),Subset(dataset = augmented_dataset, indices = augmented_test_indices)]

    test = [original_test_dataset, augmented_test_dataset]
    train = [original_train_dataset, augmented_train_dataset]

    # Concat the train/test datasets created
    train_dataset = ConcatDataset(train)
    test_dataset = ConcatDataset(test)

    return (train_dataset, test_dataset)


def load_datasets():
    print("Name of the datasets: original_dataset and augmented_dataset", )
    # Load the datasets
    if os.path.getsize("original_dataset.hickle") > 0:     
        original_dataset = hickle.load('original_dataset.hickle')
        print("original_dataset was successfully loaded!")
    else:
        original_dataset = 0
        print("original_dataset is EMPTY")

    if os.path.getsize("augmented_dataset.hickle") > 0:     
        augmented_dataset = hickle.load('augmented_dataset.hickle')
        print("augmented_dataset was successfully loaded!")
    else:
        augmented_dataset = 0
        print("augmented_dataset is EMPTY")

    return original_dataset, augmented_dataset

if __name__ == "__main__":
    original, augmented = load_datasets_hickle()    

