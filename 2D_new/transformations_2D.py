import config_2D

from torch.utils.data import Dataset
import numpy as np
import numbers
import scipy.ndimage as scp
import cv2
import random

#To perform normalization
from sklearn.preprocessing import MinMaxScaler

class ApplyNormalization(Dataset):
    """
    Applies normalization to a Dataset
    """
    
    def __init__(self, dataset, max_suvr_values = None, max_rcbf_values = None, trainingset = True):
        self.samples = []
        self.max_rcbf_values = max_rcbf_values
        self.max_suvr_values = max_suvr_values

        self.dataset = dataset
        self.trainingset = trainingset

        self._init_dataset()
        
    def _init_dataset(self):
        if self.trainingset is True:
            refImg = self.dataset.__getitem__(0)[0]

            training_images = np.zeros((len(self.dataset), np.shape(refImg)[0], np.shape(refImg)[1], np.shape(refImg)[2],  2))

            # Load all of the data to fit a MinMaxScaler
            for i in range(len(self.dataset)):
                #Load the SUVR image
                training_images[i,:,:,:,0] = self.dataset.__getitem__(i)[0][:,:,:,0]
                #Load the rCBF image
                training_images[i,:,:,:,1] = self.dataset.__getitem__(i)[0][:,:,:,1]

            # SUVr max values 0, rCBF max values 1
            max_values = np.zeros((training_images[:,:,:,:,0].shape[3], 2))
            # Find the max and min value for each slice
            for a in range(training_images[:,:,:,:,0].shape[3]):
                max_values[a,0] = training_images[:,:,:,a,0].max() 
                max_values[a,1] = training_images[:,:,:,a,1].max() 
                # Normalize both the images on ONE slice
                training_images[:,:,:,a,:] = normalize_image(training_images[:,:,:,a,:], max_values[a,0],  max_values[a,1])


            # Set the max values attribute
            self.max_suvr_values = max_values[:,0]
            self.max_rcbf_values = max_values[:,1]


           # Append the normalized data
            for i in range(len(self.dataset)):
                # Fetch the disease label
                disease = self.dataset.__getitem__(i)[1]

                #Append the preprocessed data into samples
                self.samples.append((training_images[i,:,:,:,:], disease)) 

        else:
            refImg = self.dataset.__getitem__(0)[0]

            test_images = np.zeros((len(self.dataset), np.shape(refImg)[0], np.shape(refImg)[1], np.shape(refImg)[2],  2))

            # Load all of the data to fit a MinMaxScaler
            for i in range(len(self.dataset)):
                #Load the SUVR image
                test_images[i,:,:,:,0] = self.dataset.__getitem__(i)[0][:,:,:,0]
                #Load the rCBF image
                test_images[i,:,:,:,1] = self.dataset.__getitem__(i)[0][:,:,:,1]
            
            for a in range(test_images[:,:,:,:,0].shape[3]):
                # Normalize both the images on ONE slice
                test_images[:,:,:,a,:] = normalize_image(test_images[:,:,:,a,:], self.max_suvr_values[a],  self.max_rcbf_values[a])

           # Append the normalized data
            for i in range(len(self.dataset)):
                # Fetch the disease label
                disease = self.dataset.__getitem__(i)[1]

                #Append the preprocessed data into samples
                self.samples.append((test_images[i,:,:,:,:], disease)) 
  
           

    def __getitem__(self, idx): 
        scans, disease = self.samples[idx]

        return scans, disease

    def __len__(self):
        return len(self.samples)

 # Normalizes the images according to the whole datasets mean and standard deviance
def normalize_image(images, max_value_suvr, max_value_rcbf):
    images[:,:,:,0] = (images[:,:,:,0]) / (max_value_suvr)
    images[:,:,:,1] = (images[:,:,:,1]) / (max_value_rcbf)

    return images

def getMultipleSliceImages(scans, sliceSample = None, mirror = True):  
    if sliceSample is None:
        sliceSample = config_2D.sliceSample

    if mirror is True:
        # Randomize a value between 0 and 1
        rand = random.random()
        if rand > 0.5:
            for i in range(9): #Flip the 9 images to be concenated
                scans[:,:,sliceSample - 4 + i, 0] = cv2.flip(scans[:,:,sliceSample - 4 + i, 0], 0)
                scans[:,:,sliceSample - 4 + i, 1] = cv2.flip(scans[:,:,sliceSample - 4 + i, 1], 0)

    shape = np.shape(scans)
    #Create a holder for the return images
    multipleImages = np.zeros((shape[0]*3,shape[1]*3, 2))
    
    suvr = np.block([[scans[:,:,sliceSample-4,0],scans[:,:,sliceSample-3,0], scans[:,:,sliceSample-2,0]],
                    [scans[:,:,sliceSample-1,0],scans[:,:,sliceSample,0], scans[:,:,sliceSample+1,0]],
                    [scans[:,:,sliceSample+2,0],scans[:,:,sliceSample+3,0], scans[:,:,sliceSample+4,0]]])
    
    rcbf = np.block([[scans[:,:,sliceSample-4,1],scans[:,:,sliceSample-3,1], scans[:,:,sliceSample-2,1]],
                [scans[:,:,sliceSample-1,1],scans[:,:,sliceSample,1], scans[:,:,sliceSample+1,1]],
                [scans[:,:,sliceSample+2,1],scans[:,:,sliceSample+3,1], scans[:,:,sliceSample+4,1]]])
    
    multipleImages[:,:,0] = suvr
    multipleImages[:,:,1] = rcbf



    return multipleImages

def crop_center(scans, cropx = None, cropy = None):
    if cropx is None:
        cropx = config_2D.cropx
    if cropy is None:
        cropy = config_2D.cropy

    y,x,_,_ = scans.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2  

    return scans[starty:starty + cropy, startx:startx + cropx, :]

def getMeanImages(scan, numslices, slice = 64):
    #returns mean of rcbf and suvr
    img = scan

    x,y,_,_ = img.shape
    meansuvr = np.zeros((x, y))
    meanrcbf = np.zeros((x,y))
    suvrimg = img[:,:,:,0] # takes out suvr from img
    rcbfimg = img[:,:,:,1]


    for i in range (slice-numslices, slice+numslices):
        meansuvr += suvrimg[:,:,i]
        meanrcbf += rcbfimg[:,:,i]

    meanrcbf = meanrcbf/(2*numslices)
    meansuvr = meansuvr/(2*numslices)

    ret = np.zeros((x,y,2))
    ret[:,:,0] = meansuvr
    ret[:,:,1] = meanrcbf
    
    return ret

def getMeanNormalbrain_rcbf(dataset, slicenum = 67):
    #return the mean brain, the dataset is hardcoded to be normal brains in the range 12,18
    meannormalbrain = np.zeros((128,128)) #store the mean brain of all the normal suvr brains
    counter = 0
    for i in range(len(dataset)): # iteratets through the current dataset and extracts the mean normal brain
        
        scan,label = dataset[i]
        if np.array_equal(label,[1,0,0,0]): #checks if it is a normal brain
            counter += 1
            img_suvr = scan[:,:,slicenum,0] #then retrivies normal brain
            meannormalbrain += img_suvr # annd adds
    if counter == 0:
        meannormalbrain = meannormalbrain/1
    else:
        meannormalbrain = meannormalbrain/(counter)

    return meannormalbrain

def getMeanNormalbrain_suvr(dataset, slicenum = 67):
    #return the mean brain, the dataset is hardcoded to be normal brains in the range 12,18
    meannormalbrain = np.zeros((128,128)) #store the mean brain of all the normal suvr brains
    counter = 0
    for i in range(len(dataset)): # iteratets through the current dataset and extracts the mean normal brain
        
        scan,label = dataset[i]
        if np.array_equal(label,[1,0,0,0]): #checks if it is a normal brain
            counter += 1
            img_suvr = scan[:,:,slicenum,1] #then retrivies normal brain
            meannormalbrain += img_suvr # and adds

    if counter == 0:
        meannormalbrain = meannormalbrain / 1
    else:
        meannormalbrain = meannormalbrain / (counter)

    return meannormalbrain

class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    
    def __init__(self, dataset, sliceNr = None, applyMean = False, applyDiffnormal = False, meantrainbrain_rcbf = None, meantrainbrain_suvr = None,
                useMultipleSlices = False, mirrorImage = None, gammaTransform = None, transform = None, randomSlice = False):

        self.dataset = dataset
        self.transform = transform
        self.sliceSample = sliceNr
        self.applyMean = applyMean
        self.useMultipleSlices = useMultipleSlices
        self.applyDiffnormal = applyDiffnormal
        self.applyGammaTransformation = gammaTransform
        self.applyMirrorImage = mirrorImage
        self.meannormalbrain_rcbf = meantrainbrain_rcbf
        self.meannormalbrain_suvr = meantrainbrain_suvr
        self.randomSlice = randomSlice

        if applyDiffnormal is True and meantrainbrain_rcbf is None and meantrainbrain_suvr is None :
            self.meannormalbrain_rcbf = getMeanNormalbrain_rcbf(dataset, self.sliceSample)
            self.meannormalbrain_suvr = getMeanNormalbrain_suvr(dataset, self.sliceSample)
        else:
            self.meannormalbrain_rcbf = meantrainbrain_rcbf
            self.meannormalbrain_suvr = meantrainbrain_suvr

    def __getitem__(self, idx):
        scans, disease = self.dataset[idx]

        # Set new sliceSample + [-1, 1]
        if self.randomSlice is True:
            sliceSampleNew = self.sliceSample + random.randint(-3,3)
        else:
            sliceSampleNew = self.sliceSample

        if self.applyMean is True and self.sliceSample is not None:
            shape = np.shape(scans)
            temp = np.zeros((shape[0],shape[1],shape[3]+2))
            meanImgs = getMeanImages(scans, config_2D.numberOfmeanslices, sliceSampleNew)
            temp[:,:,0] = scans[:,:,sliceSampleNew,0]
            temp[:,:,1] = scans[:,:,sliceSampleNew,1]
            temp[:,:,2] = meanImgs[:,:,0]
            temp[:,:,3] = meanImgs[:,:,1]
            scans = temp

        # Fetch the samples specified in the configuration file
        if self.applyMean is False and self.sliceSample is not None:

            if self.useMultipleSlices is True:
                cropped_scans = crop_center(scans)

                if self.applyMirrorImage is False:
                    temp = getMultipleSliceImages(cropped_scans, sliceSample = sliceSampleNew, mirror = False)
                else: 
                    temp = getMultipleSliceImages(cropped_scans, sliceSample = sliceSampleNew, mirror = True)
                
                scans = temp

            else: #else only use one slice
                temp = scans[:,:,sliceSampleNew,:]
                scans = temp
            
        # Apply transform to take the difference between meannormal brain and currentbrain    
        if self.applyDiffnormal is True and self.applyMean is True:
            # RCBF
            if self.meannormalbrain_rcbf is not None: # this is for test data
                temp[:,:,2] = np.abs(scans[:,:,0] - self.meannormalbrain_rcbf)
            
            else:
                temp[:,:,2] = np.abs(scans[:,:,0] - self.meannormalbrain_rcbf)

            # SUVR
            if self.meannormalbrain_suvr is not None: # this is for test data
                temp[:,:,3] = np.abs(scans[:,:,1] - self.meannormalbrain_suvr)
            
            else:
                temp[:,:,3] = np.abs(scans[:,:,1] - self.meannormalbrain_suvr)

        if self.applyDiffnormal is True and self.applyMean is False:
            print("Please set addMeanImage = True in the configuration file")
            exit()

        if self.applyGammaTransformation is not None:
            #rand = (random.random() * self.applyGammaTransformation) + 0.95
            #scans = np.power(scans, rand)
            scans = np.power(scans, self.applyGammaTransformation)

        if self.applyMirrorImage is True and self.useMultipleSlices is False:
            scans = mirrorImages(scans)

        # Apply transform specified in the configuration file
        if self.transform is not None:
            temp = self.transform(scans)
            scans = temp
            
        #scans = scans
        return scans, disease 

    def __len__(self):
        return len(self.dataset)


    
class resize(object):
    """ Function to resize object to given size

    return resized object

    """
    def __call__(self, img, sizetobe = 224):

        img = cv2.resize(img, dsize=(sizetobe,sizetobe))
        
        return img

def mirrorImages(images):
    # Randomize a value between 0 and 1
    rand = random.random()
    if rand > 0.5:
        images = cv2.flip(images, 0)

    return images

class mirrorImage(object):
    """ Function to mirror the object around horizontal axis

    return mirrored object

    """
    def __call__(self, images):
        # Randomize a value between 0 and 1
        rand = random.random()
        if rand > 0.5:
            images = cv2.flip(images, 0)

        return images

