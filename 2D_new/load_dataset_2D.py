#To import configuration parameters
import config_2D
# To load the datasets from .pickle files
import pickle
import os
# Needed to randomize indices
import random
# Needed to split the dataset into train/test subsets
from torch.utils.data import Dataset, Subset, ConcatDataset
# Our own files
from create_dataset_2D import ScanDataSet
#Standard libraries
import numpy as np

def load_original_dataset():
    # Load the datasets
    if os.path.getsize("original_dataset_2D.pickle") > 0:    
        pickle_in = open("original_dataset_2D.pickle","rb")
        original_dataset = pickle.load(pickle_in)       
        print("'original_dataset_2D.pickle' was successfully loaded!")
    else:
        original_dataset = 0
        print("'original_dataset_2D.pickle' is EMPTY")


    return original_dataset

if __name__ == "__main__":
    original = load_original_dataset()  



