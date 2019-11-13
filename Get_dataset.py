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

# To rotate the images
import scipy.ndimage


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
        scans = sample[0]
        disease = sample[1]

        #Find the centered pixel according to the maximum intensity pixel areas in the SUVR scan
        centered_pixel_position = preprocessing.find_centered_pixel(sample)


        #Crop the image
        cropped_scans[:,:,:,0]  = scans[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz, 0] #SUVr
        cropped_scans[:,:,:,1]  = scans[:,:, centered_pixel_position[2] - self.nrPixelsz : centered_pixel_position[2] + self.nrPixelsz, 1] #rCBF

        return (scans, disease)

#
#
# PREPROCESSING TRANSFORMS
#
#

#Extract positions over threshold
#if(bild==22 or bild ==20):
#    pixel_threshold=0.35
#else:
#    pixel_threshold=0.1



class Rotate_around_x(object):
    #
    # Rotate the scan around the x-axis
    #

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        scans = sample[0]
        disease = sample[1] 

        #Fetch reference to get the shape of the images
        shape = np.shape(scipy.ndimage.interpolation.rotate(scans[:,:,:,0], -self.degrees, axes = (1,2)))
        #Create an array to store the scans of all the patients
        rotated_scans = np.zeros((shape[0], shape[1], shape[2], 2))

        #Rotate around x-axis to align the brain with the vertical line
        rotated_scans[:,:,:,0] = scipy.ndimage.interpolation.rotate(scans[:,:,:,0], -self.degrees, axes = (1,2)) #SUVr
        rotated_scans[:,:,:,1] = scipy.ndimage.interpolation.rotate(scans[:,:,:,1], -self.degrees, axes = (1,2)) #rCBF

        return (rotated_scans, disease)

def crop_image(image, pixeltop, pixelbottom):
    #Find the centered pixel according to the maximum intensity pixel areas in the SUVR scan
    centered_pixel_position = preprocessing.find_centered_pixel(sample)

    shape = np.shape(scans[:,:, centered_pixel_position[2] - self.nrPixelsBottom : centered_pixel_position[2] + self.nrPixelsTop, 0])
    #Create an array to store the scans of all the patients
    cropped_scans = np.zeros((shape[0], shape[1], shape[2], 2))

    #Crop the image
    cropped_scans[:,:,:,0] = scans[:,:, centered_pixel_position[2] - self.nrPixelsBottom : centered_pixel_position[2] + self.nrPixelsTop, 0] #SUVr
    cropped_scans[:,:,:,1] = scans[:,:, centered_pixel_position[2] - self.nrPixelsBottom : centered_pixel_position[2] + self.nrPixelsTop, 1] #rCBF

    return cropped_image

class Crop(object):
    #
    #Crops the image in a sample
    #

    def __init__(self, nrPixelsTop = None, nrPixelsBottom = None):
        self.nrPixelsTop = nrPixelsTop
        self.nrPixelsBottom = nrPixelsBottom

    def __call__(self, sample):
        scans = sample[0]
        disease = sample[1]

        #Find the centered pixel according to the maximum intensity pixel areas in the SUVR scan
        centered_pixel_position = preprocessing.find_centered_pixel(sample)

        shape = np.shape(scans[:,:, centered_pixel_position[2] - self.nrPixelsBottom : centered_pixel_position[2] + self.nrPixelsTop, 0])
        #Create an array to store the scans of all the patients
        cropped_scans = np.zeros((shape[0], shape[1], shape[2], 2))

        #Crop the image
        cropped_scans[:,:,:,0] = scans[:,:, centered_pixel_position[2] - self.nrPixelsBottom : centered_pixel_position[2] + self.nrPixelsTop, 0] #SUVr
        cropped_scans[:,:,:,1] = scans[:,:, centered_pixel_position[2] - self.nrPixelsBottom : centered_pixel_position[2] + self.nrPixelsTop, 1] #rCBF

        return (cropped_scans, disease)


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
 
        scans, disease = self.samples[idx]
        
        return scans, disease

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
        refImg = ecat.load(listFilesECAT_SUVR[0]).get_frame(0)

        #Create an array to store the scans of all the patients
        images = np.zeros((np.shape(refImg)[0],np.shape(refImg)[1],np.shape(refImg)[2], 2))

        #Append the SUVR and rCBF images and the corresponding label into one sample
        for nr in range(np.size(listFilesECAT_SUVR)):


            #Load the SUVR image
            images[:,:,:,0] = ecat.load(listFilesECAT_SUVR[nr]).get_frame(0)

            #Load the rCBF image
            images[:,:,:,1] = ecat.load(listFilesECAT_rCBF[nr]).get_frame(0)

            # Preprocess our data


            #Append the preprocessed data into samples
            self.samples.append((images, diseases[nr,:]))

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