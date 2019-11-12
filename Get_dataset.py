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

import preprocessing


#
# DATA AUGMENTATION TRANSFORMS
#
class Random_rotate_around_z(object):
    #
    # Rotate the scan around the x-axis
    #

    def __init__(self, nrPixelsz = None):
        self.nrPixelsz = nrPixelsz

    def __call__(self, sample):
        SUVR = sample[0]
        rCBF = sample[1]
        disease = sample[2]

        #Find the centered pixel according to the maximum intensity pixel areas in the SUVR scan
        centered_pixel_position = preprocessing.find_centered_pixel(sample)

        #Crop the image
        SUVR_cropped  = SUVR[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz]
        rCBF_cropped  = rCBF[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz]

        return (SUVR_cropped, rCBF_cropped, disease)

#
#
# PREPROCESSING TRANSFORMS
#
#

class Rotate_around_x(object):
    #
    # Rotate the scan around the x-axis
    #

    def __init__(self, nrPixelsz = None):
        self.nrPixelsz = nrPixelsz

    def __call__(self, sample):
        SUVR = sample[0]
        rCBF = sample[1]
        disease = sample[2]

        #Find the centered pixel according to the maximum intensity pixel areas in the SUVR scan
        centered_pixel_position = preprocessing.find_centered_pixel(sample)

        #Crop the image
        SUVR_cropped  = SUVR[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz]
        rCBF_cropped  = rCBF[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz]

        return (SUVR_cropped, rCBF_cropped, disease)


class Crop(object):
    #
    #Crops the image in a sample
    #

    def __init__(self, nrPixelsz = None):
        self.nrPixelsz = nrPixelsz

    def __call__(self, sample):
        SUVR = sample[0]
        rCBF = sample[1]
        disease = sample[2]

        #Find the centered pixel according to the maximum intensity pixel areas in the SUVR scan
        centered_pixel_position = preprocessing.find_centered_pixel(sample)

        #Check if we will end up above or below with our specified cropping
        if(centered_pixel_position[2] - self.nrPixelsz < SUVR.shape[2]):
            SUVR_cropped  = SUVR[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz]
        else: # Assume zero padding above/below image
            SUVR_cropped = np.zeros((SUVR.shape[0], SUVR.shape[1], SUVR.shape[2]))
            SUVR_cropped  = SUVR[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz]




       if(centered_pixel_position[2] - self.nrPixelsz < rCBF.shape[2]):
            rCBF_cropped  = rCBF[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz]
        else:


        #Crop the image
        rCBF_cropped  = rCBF[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz]

        return (SUVR_cropped, rCBF_cropped, disease)


class ScanDataSet(Dataset):
    def __init__(self, image_root, label_root, filetype_SUVR, filetype_rCBF, transform = None):
        self.image_root = image_root
        self.label_root = label_root
        self.filetype_SUVR = filetype_SUVR #indicates which kind of scan, use ""
        self.filetype_rCBF = filetype_rCBF #indicates which kind of scan, use ""

        self.samples = []
        self.disease = LabelEncoder()
        self.transform = transform
        self._init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #Apply the transform to the sample if specified
        if self.transform:
            self.samples[idx] = self.transform(self.samples[idx])
 
        SUVR_scan, rCBF_scan, disease = self.samples[idx]
        
        return SUVR_scan, rCBF_scan, disease

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
        
        # *****Reads the scans *********

        #create an empty list for SUVR
        listFilesECAT_SUVR = [] 
        for dirName, subdirList, fileList in os.walk(self.image_root):
            for filename in fileList:
                if self.filetype_SUVR in filename.lower(): 
                    listFilesECAT_SUVR.append(os.path.join(dirName, filename))
        
        #create an empty list for rCBF
        listFilesECAT_rCBF = [] 
        for dirName, subdirList, fileList in os.walk(self.image_root):
            for filename in fileList:
                if self.filetype_rCBF in filename:
                    listFilesECAT_rCBF.append(os.path.join(dirName, filename))

        listFilesECAT_SUVR.sort()
        listFilesECAT_rCBF.sort()


        # Fetch a reference image to be able to create the right dimensions on the images
        refImg_SUVR = ecat.load(listFilesECAT_SUVR[0]).get_frame(0)
        refImg_rCBF = ecat.load(listFilesECAT_rCBF[0]).get_frame(0)

        #Create an array to store the scans of all the patients
        image_SUVR = np.zeros((np.shape(refImg_SUVR)[0],np.shape(refImg_SUVR)[1],np.shape(refImg_SUVR)[2]))
        image_rCBF = np.zeros((np.shape(refImg_rCBF)[0],np.shape(refImg_rCBF)[1],np.shape(refImg_rCBF)[2]))

        #Append the SUVR and rCBF images and the corresponding label into one sample
        for nr in range(np.size(listFilesECAT_SUVR)):
            #Load the SUVR image
            image_SUVR = ecat.load(listFilesECAT_SUVR[nr]).get_frame(0)
            image_SUVR = image_SUVR[...,np.newaxis]

            #Load the rCBF image
            image_rCBF = ecat.load(listFilesECAT_rCBF[nr]).get_frame(0)
            image_rCBF = image_rCBF[...,np.newaxis]

            #Append them into one sample
            self.samples.append((image_SUVR, image_rCBF, diseases[nr,:]))

""" class ScanDataSet(Dataset):
    def __init__(self, image_root, label_root, filetype_SUVR, filetype_rCBF):
        self.image_root = image_root
        self.label_root = label_root
        self.filetype_SUVR = filetype_SUVR #indicates which kind of scan, use ""
        self.filetype_rCBF = filetype_rCBF #indicates which kind of scan, use ""

        self.samples = []
        self.disease = LabelEncoder()
        self._init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        SUVR_scan, rCBF_scan, disease = self.samples[idx]
        return SUVR_scan, rCBF_scan, disease

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
        
        # *****Reads the scans *********

        #create an empty list for SUVR
        listFilesECAT_SUVR = [] 
        for dirName, subdirList, fileList in os.walk(self.image_root):
            for filename in fileList:
                if self.filetype_SUVR in filename.lower(): 
                    listFilesECAT_SUVR.append(os.path.join(dirName, filename))
        
        #create an empty list for rCBF
        listFilesECAT_rCBF = [] 
        for dirName, subdirList, fileList in os.walk(self.image_root):
            for filename in fileList:
                if self.filetype_rCBF in filename:
                    listFilesECAT_rCBF.append(os.path.join(dirName, filename))

        listFilesECAT_SUVR.sort()
        listFilesECAT_rCBF.sort()


        # Fetch a reference image to be able to create the right dimensions on the images
        refImg_SUVR = ecat.load(listFilesECAT_SUVR[0]).get_frame(0)
        refImg_rCBF = ecat.load(listFilesECAT_rCBF[0]).get_frame(0)

        #Create an array to store the scans of all the patients
        image_SUVR = np.zeros((np.shape(refImg_SUVR)[0],np.shape(refImg_SUVR)[1],np.shape(refImg_SUVR)[2]))
        image_rCBF = np.zeros((np.shape(refImg_rCBF)[0],np.shape(refImg_rCBF)[1],np.shape(refImg_rCBF)[2]))

        #Append the SUVR and rCBF images and the corresponding label into one sample
        for nr in range(np.size(listFilesECAT_SUVR)):
            #Load the SUVR image
            image_SUVR = ecat.load(listFilesECAT_SUVR[nr]).get_frame(0)
            image_SUVR = image_SUVR[...,np.newaxis]

            #Load the rCBF image
            image_rCBF = ecat.load(listFilesECAT_rCBF[nr]).get_frame(0)
            image_rCBF = image_rCBF[...,np.newaxis]

            #Append them into one sample
            self.samples.append((image_SUVR, image_rCBF, diseases[nr,:]))
 """