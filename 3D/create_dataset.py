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
        scans, disease = self.samples[idx]
        
        return scans, disease

    def _init_dataset(self):
        # ***** Read the disease labels from a csv file (saved from the patientlist in the xlms file) *****
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
        cropped_images = np.zeros((np.size(listFilesECAT_SUVR),np.shape(refImg)[0],np.shape(refImg)[1], 80, 2)) #28 x 128 x 128 x 80 x 2 - HARD CODED CROP top = 30, bottom = 50

        # Preprocess our data
        # Load the whole dataset and save into numpy array and perform cropping and rotation around x-axis
        for nr in range(np.size(listFilesECAT_SUVR)):
             #Load the SUVR image
            images[:,:,:,0] = ecat.load(listFilesECAT_SUVR[nr]).get_frame(0)
            #Load the rCBF image
            images[:,:,:,1] = ecat.load(listFilesECAT_rCBF[nr]).get_frame(0)

            # Rotate the images around the x-axis with -20 degrees
            rotated_images = rotate_image_around_x(images)
            
            # Crops the z-axis and stores all of the cropped images to be able to calculate the statistics before normalization
            cropped_images[nr,:,:,:,:] = crop_z_axis(rotated_images)


        #  Calculate statistics in order to be able to normalize the dataset
        mean_SUVR, std_SUVR, mean_rCBF, std_rCBF = calculate_mean_std(cropped_images) 

        # Normalize the data and append them into samples with the corresponding label
        for nr in range(np.size(listFilesECAT_SUVR)):
            normalized_images = normalize_image(cropped_images[nr,:,:,:,:], mean_SUVR, std_SUVR, mean_rCBF, std_rCBF)

            #Append the preprocessed data into samples
            self.samples.append((normalized_images, diseases[nr,:]))


# ********************* PREPROCESSING FUNCTIONS ****************************

# Rotates the image around x-axis with 20 degrees
def rotate_image_around_x(images, degrees = config.fixed_degrees_z):
    
    #Rotate around x-axis to align the brain with the vertical line
    images[:,:,:,0] = scipy.ndimage.interpolation.rotate(images[:,:,:,0], -degrees, axes = (1,2), reshape = False) #SUVr
    images[:,:,:,1] = scipy.ndimage.interpolation.rotate(images[:,:,:,1], -degrees, axes = (1,2), reshape = False) #rCBF

    return images

# Crops the images in the z-direction
def crop_z_axis(images, nrPixelsTop = config.nrPixelsTop, nrPixelsBottom = config.nrPixelsBottom):
    #Find the centered pixel according to the maximum intensity pixel areas in the SUVR scan
    centered_pixel_position = find_centered_pixel(images)

    shape = np.shape(images[:,:, centered_pixel_position[2] - nrPixelsBottom : centered_pixel_position[2] + nrPixelsTop, 0])
    #Create an array to store the scans of all the patients
    cropped_scans = np.zeros((shape[0], shape[1], shape[2], 2))

    #Crop the image
    cropped_scans[:,:,:,0] = images[:,:, centered_pixel_position[2] - nrPixelsBottom : centered_pixel_position[2] + nrPixelsTop, 0] #SUVr
    cropped_scans[:,:,:,1] = images[:,:, centered_pixel_position[2] - nrPixelsBottom : centered_pixel_position[2] + nrPixelsTop, 1] #rCBF

    return cropped_scans

# Normalizes the images according to the whole datasets mean and standard deviance
def normalize_image(images, mean_SUVR, std_SUVR, mean_rCBF, std_rCBF):
    images[:,:,:,0] = (images[:,:,:,0] - mean_SUVR) / std_SUVR
    images[:,:,:,1] = (images[:,:,:,1] - mean_rCBF) / std_rCBF

    return images

# Calculates the mean and standard deviance of the whole dataset
def calculate_mean_std(images):
    #return (images[:,:,:,:,0].mean(), np.sqrt(images[:,:,:,:,0].var()), images[:,:,:,:,1].mean(), np.sqrt(images[:,:,:,:,1].var()))
    return (images[np.nonzero(images[:,:,:,:,0])].mean(), np.sqrt(images[np.nonzero(images[:,:,:,:,0])].var()), images[np.nonzero(images[:,:,:,:,1])].mean(), np.sqrt(images[np.nonzero(images[:,:,:,:,1])].var()))


# ********************* AUGMENTATION FUNCTIONS *****************************

# Applies Gaussian distributed random noise to the dataset
def noise_dataset(original_dataset):
    #Fetch reference sample in order to create matrix to be able to save all of the images
    scan, dis = original_dataset[0]
    shape = scan.shape
    noisenp = np.zeros((len(original_dataset),shape[0],shape[1],shape[2],2))
    labelsnp = np.zeros((len(original_dataset),dis.shape[0]))

    for i in range(len(original_dataset)):
        randomNoise = np.random.rand(shape[0], shape[1], shape[2], 2)
        scan,dis = original_dataset[i]
        noisenp[i,:,:,:,:] = np.add(scan, randomNoise) # Adding random noise from the standard nornmal dist
        labelsnp[i,:] = dis

    # Casts the numpy array to the class "Dataset"
    noise_dataset  = numpydataset.numpy_to_dataset(noisenp,labelsnp)
    return noise_dataset

# Rotates the datasets randomly with 20 degrees around z-axis, 5 degrees around x-axis
def rotate_dataset_around_z(original_dataset, degree_x = config.random_degrees_x, degree_z = config.random_degrees_z):
    #Fetch reference sample
    scan, dis = original_dataset[0]
    shape = scan.shape
    rotatep = np.zeros((len(original_dataset),shape[0],shape[1],shape[2],2))
    labelsnp = np.zeros((len(original_dataset),dis.shape[0]))


    for i in range(len(original_dataset)):
        scan, dis = original_dataset[i]
        rand_rot_z = np.random.uniform(-degree_x, degree_x)
        rand_rot_x = np.random.uniform(-degree_z, degree_z)  
        rotatep[i,:,:,:,0] = scipy.ndimage.interpolation.rotate(scan[:,:,:,0], rand_rot_z, axes = (0,1), reshape = False) #SUVr
        rotatep[i,:,:,:,1] = scipy.ndimage.interpolation.rotate(scan[:,:,:,1], rand_rot_z, axes = (0,1), reshape = False) #rCBF
        rotatep[i,:,:,:,0] = scipy.ndimage.interpolation.rotate(rotatep[i,:,:,:,0], rand_rot_x, axes = (1,2), reshape = False) #SUVr
        rotatep[i,:,:,:,1] = scipy.ndimage.interpolation.rotate(rotatep[i,:,:,:,1], rand_rot_x, axes = (1,2), reshape = False) #rCBF 
        labelsnp[i,:] = dis

    rotate_dataset  = numpydataset.numpy_to_dataset(rotatep,labelsnp)
    return rotate_dataset



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

    # ****** CREATE AUGMENTED DATASETS ********
    print("Augmenting the data")
    print("Number of augmentation datasets to be created: ", config.NumberOfAugmentations)
    datasets = []
    # For number of times we want to duplicate our dataset
    for i in range(config.NumberOfAugmentations):
        dataset_temp = noise_dataset(original_dataset)
        dataset_temp = rotate_dataset_around_z(dataset_temp)

        datasets.append(dataset_temp)
        print(i, "...")

    print("Done\n")

    # Concat the datasets into one
    augmented_dataset = ConcatDataset(datasets)

    """
    hickle.dump(augmented_dataset, 'augmented_dataset.hickle', mode = 'w')
    hickle.dump(original_dataset, 'original_dataset.hickle', mode = 'w')
    """
    

    print("Saving the files into folders")

    # Save the original preprocessed dataset 
    pickle_out2 = open("original_dataset.pickle","wb")
    pickle.dump(original_dataset, pickle_out2)
    pickle_out2.close()

    # Save the concated augmented dataset
    pickle_out = open("augmented_dataset.pickle","wb")
    pickle.dump(augmented_dataset, pickle_out)
    pickle_out.close()

    print("Done")
    

   