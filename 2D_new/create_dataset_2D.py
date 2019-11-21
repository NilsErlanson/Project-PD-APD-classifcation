#
# Script which contains the function necessary load the dataset, preprocess it and augment it. 
# The main-function saves the created datasets into .pickle files that can be loaded from load_dataset.py
#

# To import configuration parameters
import config_2D
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
#import numpydataset


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
            letter = [0 for _ in range(0,config_2D.nrOfDifferentDiseases)]
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

        images = np.zeros((np.shape(refImg)[0],np.shape(refImg)[1],np.shape(refImg)[2], 2)) #28 x 128 x 128 x 128 x 2

        # Preprocess our data
        # Load the whole dataset and save into numpy array and perform cropping and rotation around x-axis
        for nr in range(np.size(listFilesECAT_SUVR)):
             #Load the SUVR image
            images[:,:,:,0] = ecat.load(listFilesECAT_SUVR[nr]).get_frame(0)
            #Load the rCBF image
            images[:,:,:,1] = ecat.load(listFilesECAT_rCBF[nr]).get_frame(0)

            # Calculate the min and max values for each image
            min_value_suvr, max_value_suvr = images[:,:,:,0].min(), images[:,:,:,0].max()
            min_value_rcbf, max_value_rcbf = images[:,:,:,1].min(), images[:,:,:,1].max()

            # Normalize the image
            normalized_images = normalize_image(images, min_value_suvr, max_value_suvr, min_value_rcbf, max_value_rcbf)

            #Append the preprocessed data into samples
            self.samples.append((normalized_images, diseases[nr,:]))


# Normalizes the images according to the whole datasets mean and standard deviance
def normalize_image(images, min_value_suvr, max_value_suvr, min_value_rcbf, max_value_rcbf):
    images[:,:,:,0] = (images[:,:,:,0] - min_value_suvr) / (-min_value_suvr + max_value_suvr)
    images[:,:,:,1] = (images[:,:,:,1] - min_value_rcbf) / (-min_value_rcbf + max_value_rcbf)

    #images[:,:,:,0] = (images[:,:,:,0] - mean_SUVR) / std_SUVR
    #images[:,:,:,1] = (images[:,:,:,1] - mean_rCBF) / std_rCBF

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
    image_root = config_2D.image_root
    label_root = config_2D.label_root
    filetype_SUVR = config_2D.filetype_SUVR
    filetype_rCBF = config_2D.filetype_rCBF

    #Load the original dataset
    print("Preprocessing the data ...")
    original_dataset_2D = ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF)
    print("Done\n")

    # Save the original preprocessed dataset 
    print("Saving the dataset into file...")
    pickle_out2 = open("original_dataset_2D.pickle","wb")
    pickle.dump(original_dataset_2D, pickle_out2)
    pickle_out2.close()

    print("Done")

    
    

   