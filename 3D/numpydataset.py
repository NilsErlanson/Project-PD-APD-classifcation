import numpy as np
import torch 
from torch.utils.data import Dataset

class numpy_to_dataset(Dataset):

    def __init__(self,matrix,labels):
        self.samples = []
        row,col = labels.shape
        for i in range(row):

            self.samples.append((matrix[i,:,:,:,:],labels[i,:]))
  
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        scans, disease = self.samples[idx]
        return scans, disease






        


        
