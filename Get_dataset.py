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

import numpydataset



#
# DATA AUGMENTATION TRANSFORMS
#


#
#
# PREPROCESSING TRANSFORMS
#
#

def rotate_image_around_x(images, degrees = 20):
    #Fetch reference to get the shape of the images
    #shape = np.shape(scipy.ndimage.interpolation.rotate(images[:,:,:,0], -degrees, axes = (1,2)), reshape = False)

    #Create an array to store the scans of all the patients
    #rotated_scans = np.zeros((shape[0], shape[1], shape[2], 2))



    #Rotate around x-axis to align the brain with the vertical line
    #rotated_scans[:,:,:,0] = scipy.ndimage.interpolation.rotate(images[:,:,:,0], -degrees, axes = (1,2), reshape = False) #SUVr
    #rotated_scans[:,:,:,1] = scipy.ndimage.interpolation.rotate(images[:,:,:,1], -degrees, axes = (1,2), reshape = False) #rCBF
    
    #Rotate around x-axis to align the brain with the vertical line
    images[:,:,:,0] = scipy.ndimage.interpolation.rotate(images[:,:,:,0], -degrees, axes = (1,2), reshape = False) #SUVr
    images[:,:,:,1] = scipy.ndimage.interpolation.rotate(images[:,:,:,1], -degrees, axes = (1,2), reshape = False) #rCBF

    return images

def crop_image(images, nrPixelsTop = 30, nrPixelsBottom = 50):
    #Find the centered pixel according to the maximum intensity pixel areas in the SUVR scan
    centered_pixel_position = preprocessing.find_centered_pixel(images)

    shape = np.shape(images[:,:, centered_pixel_position[2] - nrPixelsBottom : centered_pixel_position[2] + nrPixelsTop, 0])
    #Create an array to store the scans of all the patients
    cropped_scans = np.zeros((shape[0], shape[1], shape[2], 2))

    #Crop the image
    cropped_scans[:,:,:,0] = images[:,:, centered_pixel_position[2] - nrPixelsBottom : centered_pixel_position[2] + nrPixelsTop, 0] #SUVr
    cropped_scans[:,:,:,1] = images[:,:, centered_pixel_position[2] - nrPixelsBottom : centered_pixel_position[2] + nrPixelsTop, 1] #rCBF

    return cropped_scans

def noise_dataset(original_dataset):
    #Fetch reference sample
    scan, dis = original_dataset[0]
    shape = scan.shape
    noisenp = np.zeros((len(original_dataset),shape[0],shape[1],shape[2],2))
    labelsnp = np.zeros((len(original_dataset),dis.shape[0]))
    print(noisenp.shape)
    print(labelsnp.shape)

    for i in range(len(original_dataset)):
        randomNoise = np.random.rand(shape[0], shape[1], shape[2], 2)
        scan,dis = original_dataset[i]
        noisenp[i,:,:,:,:] = np.add(scan,randomNoise) # Adding random noise from the standard nornmal dist
        labelsnp[i,:] = dis

    noise_dataset  = numpydataset.numpy_to_dataset(noisenp,labelsnp)
    return noise_dataset

def rotate_dataset(original_dataset):
    #Fetch reference sample
    scan, dis = original_dataset[0]
    shape = scan.shape
    rotatep = np.zeros((len(original_dataset),shape[0],shape[1],shape[2],2))
    labelsnp = np.zeros((len(original_dataset),dis.shape[0]))
    print(rotatep.shape)
    print(labelsnp.shape)

    for i in range(len(original_dataset)):
        scan,dis = original_dataset[i]
        rand_rot_z=np.random.uniform(-20,20)
        rand_rot_x=np.random.uniform(-5,5)  
        rotatep[i,:,:,:,0] = scipy.ndimage.interpolation.rotate(scan[:,:,:,0], rand_rot_z, axes = (0,1), reshape = False) #SUVr
        rotatep[i,:,:,:,1] = scipy.ndimage.interpolation.rotate(scan[:,:,:,1], rand_rot_z, axes = (0,1), reshape = False) #rCBF
        rotatep[i,:,:,:,0] = scipy.ndimage.interpolation.rotate(rotatep[i,:,:,:,0], rand_rot_x, axes = (1,2), reshape = False) #SUVr
        rotatep[i,:,:,:,1] = scipy.ndimage.interpolation.rotate(rotatep[i,:,:,:,1], rand_rot_x, axes = (1,2), reshape = False) #rCBF 
        labelsnp[i,:] = dis
    rotate_dataset  = numpydataset.numpy_to_dataset(rotatep,labelsnp)
    return rotate_dataset

def normalize_image(images, mean_SUVR, std_SUVR, mean_rCBF, std_rCBF):
    images[:,:,:,0] = (images[:,:,:,0] - mean_SUVR) / std_SUVR
    images[:,:,:,1] = (images[:,:,:,1] - mean_rCBF) / std_rCBF

    return images

def calculate_mean_var(images):
    #return (images[:,:,:,:,0].mean(), np.sqrt(images[:,:,:,:,0].var()), images[:,:,:,:,1].mean(), np.sqrt(images[:,:,:,:,1].var()))
    return (images[np.nonzero(images[:,:,:,:,0])].mean(), np.sqrt(images[np.nonzero(images[:,:,:,:,0])].var()), images[np.nonzero(images[:,:,:,:,1])].mean(), np.sqrt(images[np.nonzero(images[:,:,:,:,1])].var()))


class ScanDataSet(Dataset):
    def __init__(self, image_root, label_root, filetype_SUVR, filetype_rCBF):
        self.image_root = image_root
        self.label_root = label_root
        self.filetype_SUVR = filetype_SUVR #indicates which kind of scan, use ""
        self.filetype_rCBF = filetype_rCBF #indicates which kind of scan, use ""

        self.samples = []
        self.disease = LabelEncoder()
        self.labelName = []
        #self.transform = transform
        self._init_dataset()

    def __len__(self):
        return len(self.samples)
    
    def __getName__(self,idx):
        return self.labelName[idx]

    def __getitem__(self, idx):
        #Apply the transform to the sample if specified
        #if self.transform:
        #    self.samples[idx] = self.transform(self.samples[idx])
 
        scans, disease = self.samples[idx]
        
        return scans, disease

    def _init_dataset(self):
        #Read the disaese
        tmp_df = pd.read_csv(self.label_root)

        self.labelName = tmp_df.Disease

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
        images = np.zeros((np.shape(refImg)[0],np.shape(refImg)[1],np.shape(refImg)[2], 2)) #28 x 128 x 128 x 128 x 2

        cropped_images = np.zeros((np.size(listFilesECAT_SUVR),np.shape(refImg)[0],np.shape(refImg)[1], 80, 2)) #28 x 128 x 128 x 80 x 2 - HARD CODED CROP top = 30, bottom = 50

        # Preprocess our data
        print("Preprocessing the data ...")
        # Load the whole dataset and save into numpy array and perform cropping and rotation around x-axis
        for nr in range(np.size(listFilesECAT_SUVR)):
             #Load the SUVR image
            images[:,:,:,0] = ecat.load(listFilesECAT_SUVR[nr]).get_frame(0)
            #Load the rCBF image
            images[:,:,:,1] = ecat.load(listFilesECAT_rCBF[nr]).get_frame(0)

            rotated_images = rotate_image_around_x(images)
            
            #Store all of the cropped images to be able to calculate the statistics before normalization
            cropped_images[nr,:,:,:,:] = crop_image(rotated_images)
            self.samples.append((cropped_images[nr,:,:,:,:], diseases[nr,:]))


        #  Calculate statistics in order to be able to normalize the dataset
       # mean_SUVR, std_SUVR, mean_rCBF, std_rCBF = calculate_mean_var(cropped_images) 

        # Normalize the data and append them into samples with the corresponding label
        #for nr in range(np.size(listFilesECAT_SUVR)):
        #    normalized_images = normalize_image(cropped_images[nr,:,:,:,:], mean_SUVR, std_SUVR, mean_rCBF, std_rCBF)

            #Append the preprocessed data into samples
            #self.samples.append((images, diseases[nr,:]))

        print("Done")

"""
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
"""

"""
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
"""

"""
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
"""

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

 