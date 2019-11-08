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
import Get_dataset
import visFuncs


def read_data():
    data_path = 'projectfiles_PE2I/scans/ecat_scans'

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
        df = pd.read_csv('projectfiles_PE2I/patientlist.csv',na_values= '?')
        print(df.Label)
        labels = df.Label.astype(np.int64) #Integer labels
        one_hot_encode = list()
        for value in labels:
            letter = [0 for _ in range(0,6)]
            letter[value] = 1
            one_hot_encode.append(letter)
        

        self.labels = np.array(one_hot_encode)
        self.samples = read_data() #Better to let samples be the pathway, more memory efficiency
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        X = self.samples[index]
        Y = self.labels[index]
        return X,Y

def network():
    from keras.layers import Dense, Activation, Flatten

    model = keras.models.Sequential()

    model.add(Conv3D(1, kernel_size=(3,3,3), input_shape = (128, 128, 128, 1)))
    model.add(Flatten())
    model.add(Dense(6,input_dim = 126*126*126))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def resNet():
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    return resnet18

if __name__ == "__main__":
    from keras.utils import to_categorical
    image_root = 'projectfiles_PE2I/scans/ecat_scans/'
    label_root = 'projectfiles_PE2I/patientlist.csv'
    filetype = "1.v"
   # dataset = Get_dataset.ScanDataSet(image_root,label_root,filetype)  
    dataset  = ImgDataset()
    # sample = dataset.__getitem__(5)
    # visFuncs.show_scan(sample)
    # visFuncs.scroll_slices(sample)
    indices = [0,5,12,18,23]
    test_data = dataset[indices]
    train_data = dataset[[1,2,3,4,6,7,8,9,10,11,14,15,16,17,19,20,21,22,24,25,26,27]]

    x_train = train_data[:][0]
    y_train = train_data[:][1]
    print(y_train.shape)

    x_test = test_data[:][0]
    y_test = test_data[:][1]
    #y_test = to_categorical(y_test)
    print(y_test.shape)
    CNN = network()
    CNN.fit(x_train,y_train,validation_data=(x_test,y_test),epochs  = 3)

    pass