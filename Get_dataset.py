#Script for getting the the scans and diseases into a torch dataset
#Author: Nissnessthebuisness and LyckTheMyck
import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from nibabel import ecat


class ScanDataSet(Dataset):
    def __init__(self, image_root,label_root,filetype):
        self.image_root = image_root
        self.label_root = label_root
        self.filetype = filetype #indicates which kind of scan, use ""
        self.samples = []
        
        self.disease = LabelEncoder()
        self._init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scan, disease = self.samples[idx]
        return scan,disease

    def _init_dataset(self):
        
        #Read the disaese
        tmp_df = pd.read_csv(self.label_root)
        
        labels = tmp_df.Label.astype(np.int64) #Integer labels
        one_hot_encode = list()
        for value in labels:
            letter = [0 for _ in range(0,6)]
            letter[value] = 1
            one_hot_encode.append(letter)
        diseases = np.array(one_hot_encode)
        self.disease = diseases
        
        #Reads the scans
        listFilesECAT = [] #create an empty list
        for dirName, subdirList, fileList in os.walk(self.image_root):
            for filename in fileList:
                if self.filetype in filename.lower(): #check wheter the file's ECAT
                    listFilesECAT.append(os.path.join(dirName, filename))
        
        listFilesECAT.sort()

        refImg = ecat.load(listFilesECAT[0]).get_frame(0)

        #Create an array to store the scans of all the patients
        images = np.zeros((np.shape(refImg)[0],np.shape(refImg)[1],np.shape(refImg)[2]))
        images = images[...,np.newaxis]
        for nr in range(np.size(listFilesECAT)):
            images = ecat.load(listFilesECAT[nr]).get_frame(0)
            images = images[...,np.newaxis]
            self.samples.append((images,diseases[nr,:]))
                

                
    