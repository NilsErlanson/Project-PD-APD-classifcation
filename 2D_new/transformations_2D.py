import config_2D

from torch.utils.data import Dataset
import numpy as np
import numbers
import scipy.ndimage as scp
import cv2
import random

def getMultipleSliceImages(scans, sliceSample = None):  
    if sliceSample is None:
        sliceSample = config_2D.sliceSample

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

def getMeanNormalbrain(dataset, slicenum = 64):
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

class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    
    def __init__(self, dataset, sliceNr = None, applyMean = False, applyDiffnormal = False, meantrainbrain= None, 
                 useMultipleSlices = False, gammaTransform = None, transform = None):
        self.dataset = dataset
        self.transform = transform

        # Randomize a slice from -3 to + 3 from the given sliceNr
        if config_2D.sliceSample is None:
            sliceNr = 64

        self.sliceSample = sliceNr

        self.applyMean = applyMean
        self.useMultipleSlices = useMultipleSlices
        self.applyDiffnormal = applyDiffnormal
        self.applyGammaTransformation = gammaTransform
        self.meantrainbrain = meantrainbrain

        if applyDiffnormal is True:
            self.meannormalbrain = getMeanNormalbrain(dataset, self.sliceSample)
        else:
            self.meannormalbrain = None

    def __getitem__(self, idx):
        scans, disease = self.dataset[idx]

        # Set new sliceSample + [-3, 3]
        sliceSampleNew = self.sliceSample + random.randint(-1,1)

        if self.applyMean is True and sliceSampleNew is not None:
            shape = np.shape(scans)
            temp = np.zeros((shape[0],shape[1],shape[3]+2))
            meanImgs = getMeanImages(scans, config_2D.numberOfmeanslices, sliceSampleNew)
            temp[:,:,0] = scans[:,:,sliceSampleNew,0]
            temp[:,:,1] = scans[:,:,sliceSampleNew,1]
            temp[:,:,2] = meanImgs[:,:,0]
            temp[:,:,3] = meanImgs[:,:,1]
            scans = temp

        # Fetch the samples specified in the configuration file
        if self.applyMean is False and sliceSampleNew is not None:

            if self.useMultipleSlices is True:
                cropped_scans = crop_center(scans)
                temp = getMultipleSliceImages(cropped_scans)
                scans = temp
            else: #else only use one slice
                temp = scans[:,:,sliceSampleNew,:]
                scans = temp
            
        # Apply transform to take the difference between meannormal brain and currentbrain    
        if self.applyDiffnormal is True:
            if self.meantrainbrain is not None: # this is for test data
                scans[:,:,2] = np.abs(scans[:,:,0]- self.meantrainbrain)
            else:
                scans[:,:,2] = np.abs(scans[:,:,0] - self.meannormalbrain)

        if self.applyGammaTransformation is not None:
            rand = (random.random() * self.applyGammaTransformation) + 0.95
            scans = np.power(scans, rand)
            
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

