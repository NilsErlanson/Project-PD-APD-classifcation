# To load the datasets from .pickle files
import pickle
import os
# Our own files
from create_dataset_2D import ScanDataSet

def load_original_dataset():
    print("Loading dataset without transforms...")

    # Load the datasets
    if os.path.getsize("original_dataset_2D.pickle") > 0:    
        pickle_in = open("original_dataset_2D.pickle","rb")
        original_dataset = pickle.load(pickle_in)       
        print("Done!")
    else:
        original_dataset = 0
        print("Failed to load 'original_dataset_2D.pickle'")

    return original_dataset

if __name__ == "__main__":
    original_dataset = load_original_dataset()  



