#
# Script which contains the function necessary load the dataset, preprocess it and augment it. 
# The main-function saves the created datasets into .pickle files that can be loaded from load_dataset.py
#

# To import configuration parameters
import config
#Loading/Writing files
import pickle
# To load the ecat-files from folder
import os
from nibabel import ecat
# To read .csv files
import pandas as pd
# Standard scientific computing package
import numpy as np
# To create one_hot_encoded labels from the diseases
from sklearn.preprocessing import LabelEncoder
# To create the torchs Datasets 
from torch.utils.data import Dataset, ConcatDataset
# To rotate the images
import scipy.ndimage
# Our own files
import numpydataset
# To be able to generate a random number
import random

class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    #def __init__(self, dataset, sliceNr = None, applyMean = False, useMultipleSlices = False, transform = None):
    
    def __init__(self, dataset, gammaTransform = None):
        self.dataset = dataset
        self.applyGammaTransformation = gammaTransform

    def __getitem__(self, idx):
        scans, disease = self.dataset[idx]
    
        if self.applyGammaTransformation is not None:
            rand = (random.random() * self.applyGammaTransformation) + 0.95
            scans = np.power(scans, rand)
            #scans = np.power(scans, self.applyGammaTransformation)
            
        return scans, disease 

    def __len__(self):
        return len(self.dataset)


class ScanDataSet(Dataset):
    def __init__(self, image_root, label_root, filetype_SUVR, filetype_rCBF):
        self.image_root = image_root
        self.label_root = label_root
        self.filetype_SUVR = filetype_SUVR #indicates which kind of scan, use ""
        self.filetype_rCBF = filetype_rCBF #indicates which kind of scan, use ""

        self.samples = []
        self.disease = LabelEncoder()
        self.labelName = []
        self._init_dataset()

    def __len__(self):
        return len(self.samples)
    
    def __getName__(self,idx):
        return self.labelName[idx]

    def __getitem__(self, idx): 
        scans, disease = self.samples[idx]
        
        return scans, disease

    def _init_dataset(self):
        # ***** Read the disease labels from a csv file (saved from the patientlist in the xlms file) *****
        tmp_df = pd.read_csv(self.label_root)

        self.labelName = tmp_df.Disease

        labels = tmp_df.Label.astype(np.int64) #Integer labels
        one_hot_encode = list()
        for value in labels:
            letter = [0 for _ in range(0, config.nrOfDifferentDiseases)]
            letter[value] = 1
            one_hot_encode.append(letter)

        diseases = np.array(one_hot_encode)

        self.disease = diseases
        
        # ***** Reads the scans ******

        # Create an empty list for SUVR
        listFilesECAT_SUVR = [] 
        for dirName, subdirList, fileList in os.walk(self.image_root):
            for filename in fileList:
                if self.filetype_SUVR in filename.lower(): 
                    listFilesECAT_SUVR.append(os.path.join(dirName, filename))
        
        # Create an empty list for rCBF
        listFilesECAT_rCBF = [] 
        for dirName, subdirList, fileList in os.walk(self.image_root):
            for filename in fileList:
                if self.filetype_rCBF in filename:
                    listFilesECAT_rCBF.append(os.path.join(dirName, filename))

        # Sort the lists according to the patientnumbers
        listFilesECAT_SUVR.sort()
        listFilesECAT_rCBF.sort()

        # Fetch a reference image to be able to create the right dimensions on the images
        refImg = ecat.load(listFilesECAT_SUVR[0]).get_frame(0)

        #Create an array to store the scans of all the patients
        images = np.zeros((np.shape(refImg)[0],np.shape(refImg)[1],np.shape(refImg)[2], 2)) #28 x 128 x 128 x 128 x 2
        cropped_images_z = np.zeros((np.size(listFilesECAT_SUVR),np.shape(refImg)[0],np.shape(refImg)[1], config.nrPixelsTop +config.nrPixelsBottom, 2)) #28 x 128 x 128 x 80 x 2 - HARD CODED CROP top = 30, bottom = 50
        
        cropped_images_xy = np.zeros((np.size(listFilesECAT_SUVR), config.cropx, config.cropx, config.nrPixelsTop + config.nrPixelsBottom, 2)) 

        # Preprocess our data
        # Load the whole dataset and save into numpy array and perform cropping and rotation around x-axis
        for nr in range(np.size(listFilesECAT_SUVR)):
             #Load the SUVR image
            images[:,:,:,0] = ecat.load(listFilesECAT_SUVR[nr]).get_frame(0)
            #Load the rCBF image
            images[:,:,:,1] = ecat.load(listFilesECAT_rCBF[nr]).get_frame(0)
            
            # Crops the z-axis and stores all of the cropped images to be able to calculate the statistics before normalization
            cropped_images_z = crop_z_axis(images)
            
            # Crops the x and y-axis and stores all of the cropped images to be able to calculate the statistics before normalization
            cropped_images_xy = crop_center(cropped_images_z)

            min_value_suvr, max_value_suvr = cropped_images_xy[:,:,:,0].min(), cropped_images_xy[:,:,:,0].max()
            min_value_rcbf, max_value_rcbf = cropped_images_xy[:,:,:,1].min(), cropped_images_xy[:,:,:,1].max()

            # Normalize the image
            image = normalize_image(cropped_images_xy, min_value_suvr, max_value_suvr, min_value_rcbf, max_value_rcbf)

            #normalize the data in the range [0,1]
            self.samples.append((image, diseases[nr,:]))


# ********************* PREPROCESSING FUNCTIONS ****************************
def crop_center(scans, cropx = None, cropy = None):
    if cropx is None:
        cropx = config.cropx
    if cropy is None:
        cropy = config.cropy

    x,y,_,_ = scans.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2  
    
    return scans[startx:startx + cropx, starty:starty + cropy, :, :]

# Crops the images in the z-direction
def crop_z_axis(images, nrPixelsTop = config.nrPixelsTop, nrPixelsBottom = config.nrPixelsBottom):
    #Find the centered pixel according to the maximum intensity pixel areas in the SUVR scan
    centered_pixel_position = find_centered_pixel(images)

    centered_pixel_position = 64 # cropparound middle

    shape = np.shape(images[:,:, centered_pixel_position - nrPixelsBottom : centered_pixel_position + nrPixelsTop, 0])
    #Create an array to store the scans of all the patients
    cropped_scans = np.zeros((shape[0], shape[1], shape[2], 2))

    #Crop the image
    cropped_scans[:,:,:,0] = images[:,:, centered_pixel_position - nrPixelsBottom : centered_pixel_position + nrPixelsTop, 0] #SUVr
    cropped_scans[:,:,:,1] = images[:,:, centered_pixel_position - nrPixelsBottom : centered_pixel_position + nrPixelsTop, 1] #rCBF

    return cropped_scans

# Normalizes the images according to the whole datasets mean and standard deviance

def normalize_image(images, min_value_suvr, max_value_suvr, min_value_rcbf, max_value_rcbf):
    images[:,:,:,0] = (images[:,:,:,0] - min_value_suvr) / (-min_value_suvr + max_value_suvr)
    images[:,:,:,1] = (images[:,:,:,1] - min_value_rcbf) / (-min_value_rcbf + max_value_rcbf)

    return images

# ************************* HELP FUNCTIONS ********************************
def find_centered_pixel(images):
    # Find the pixels values which fulfills the specified criteria
    val = np.max(images[:,:,0:100,0]) * 0.65 
    # Find the position of the pixels which fulfills the criteria
    pixel_position  = np.where( images[:,:,0:100,0] > val )
    # Exctract the positions x,y,z 
    x = pixel_position[0][:]
    y = pixel_position[1][:]
    z = pixel_position[2][:]
    #Get middle position of the two high intensity areas in the brain, locate x-position
    threshold = (np.max(x) + np.min(x)) / 2
    index = x <= threshold
    xmid = (np.mean(x[index])+np.mean(x[~index]))/2
    ymid = np.mean(y).astype(int)
    zmid = np.mean(z).astype(int)

    #Find the position between the high intensity areas in the brain
    position = np.array([xmid,ymid,zmid]).astype(int)

    return position


# ************************** MAIN FUNCTION **********************************
if __name__ == "__main__":
    # Import the paths to the folders and specify filenames
    image_root = config.image_root
    label_root = config.label_root
    filetype_SUVR = config.filetype_SUVR
    filetype_rCBF = config.filetype_rCBF

    #Load the original dataset
    print("Preprocessing the data ...")
    original_dataset = ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF)
    print("Done\n")

    print("Saving the files into folders")

    # Save the original preprocessed dataset 
    pickle_out2 = open("original_dataset.pickle","wb")
    pickle.dump(original_dataset, pickle_out2)
    pickle_out2.close()

    print("Done")
    

   