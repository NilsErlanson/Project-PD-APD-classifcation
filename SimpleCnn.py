import os
import nibabel as nib
nibabel_dir = os.path.dirname(nib.__file__)
from nibabel import ecat
import pandas as pd
import torch
from torch.utils.data import Dataset
import keras

from keras.layers import Conv3D
import numpy as np
import sklearn


def read_data():
    data_path = 'Projektarbete_PE2I/Parameterbilder/ecat'

    #Collect all ecat images
    patientList = [] #create an empty list
    for dirName, subdirList, fileList in os.walk(data_path):
        for filename in fileList:
            if "1.v" in filename.lower(): #check wheter the file's ECAT
                patientList.append(os.path.join(dirName, filename))


    # List of the path to the files in the specified directory
    patientList.sort() #SUVr 1st, rCBF 2nd
    # Get reference picture to get the proper sizes
    refImg = ecat.load(patientList[0]).get_frame(0)

    #Create an array to store the scans of all the patients
    images = np.zeros(( np.size(patientList),np.shape(refImg)[0],np.shape(refImg)[1],np.shape(refImg)[2]))
    for nr in range(np.size(patientList)):
        images[nr,:,:,:] = ecat.load(patientList[nr]).get_frame(0)
    images = images[...,np.newaxis]
    return images

class ImgDataset(Dataset):
    def __init__(self):
        df = pd.read_csv('Projektarbete_PE2I/Patientlista-avid.csv')
        self.labels = df.Label
        self.samples = read_data() #Better to let samples be the pathway, more memory efficiency
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        X = self.samples[index]
        Y = self.labels[index]
        return X,Y

def network():
    model = keras.models.Sequential()

    model.add(Conv3D(1, kernel_size=(3,3,3), input_shape = (128, 128, 128, 1)))



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    dataset = ImgDataset()
    indices = [0,5,12,18,23]
    test_data = dataset[indices]
    train_data = dataset[~indices]

    x_train = train_data[:][0]
    y_train = train_data[:][1]

    x_test = test_data[:][0]
    y_test = test_data[:][1]

    CNN = network()

    pass