import config_2D
import numpy as np
import random
import numbers


from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION

import torchvision.transforms.functional as F

import skimage

import skimage.transform as ST

import scipy.ndimage as scp

import os
import cv2

 # ***************** NYTT ******************************
def getMeanImages(scan, numslices, slice = 64):
    #returns mean of rcbf and suvr
    img = scan

    x,y,_,_ = img.shape
    meansuvr = np.zeros((x, y))
    meanrcbf = np.zeros((x,y))
    suvrimg = img [:,:,:,0] # takes out suvr from img
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

def getMeanNormalbrain(dataset,slicenum = 64):
    #return the mean brain, the dataset is hardcoded to be normal brains in the range 12,18
    meannormalbrain = np.zeros((128,128)) #store the mean brain of all the normal suvr brains
    for i in range(12,18):
        scan,_ = dataset[i]
        img_suvr = scan[:,:,slicenum,0]
        meannormalbrain += img_suvr

    meannormalbrain = meannormalbrain/(18-12)
    return meannormalbrain
    

class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    
    def __init__(self, dataset, sliceNr = None, applyMean = False,normalbrain = False, transform = None):
        self.dataset = dataset
        self.transform = transform
        self.sliceSample = sliceNr
        self.applyMean = applyMean
        self.applyNormalbrain = normalbrain
        if normalbrain is True:
            self.meannormalbrain  = getMeanNormalbrain(dataset)
        
    

    def __getitem__(self, idx):
        scans, disease = self.dataset[idx]
        
        if self.applyMean is True and self.sliceSample is not None:
            shape = np.shape(scans)
            temp = np.zeros((shape[0],shape[1],shape[3]+2))
            meanImgs = getMeanImages(scans, config_2D.numberOfmeanslices, config_2D.sliceSample)
            temp[:,:,0] = scans[:,:,self.sliceSample,0]
            temp[:,:,1] = scans[:,:,self.sliceSample,1]
            temp[:,:,2] = meanImgs[:,:,0]
            temp[:,:,3] = meanImgs[:,:,1]
            scans = temp

        # Fetch the samples specified in the configuration file
        if self.applyMean is False and self.sliceSample is not None:
            temp = scans[:,:,self.sliceSample,:]
            scans = temp
        # Apply transform to take the difference between meannormal brain and currentbrain    
        if self.applyNormalbrain is True:
            scans[:,:,2] = np.abs(scans[:,:,0]-self.meannormalbrain)
            
        # Apply transform specified in the configuration file
        if self.transform is not None:
            temp = self.transform(scans)
            scans = temp

            
        scans = scans
        return scans, disease 

    def __len__(self):
        return len(self.dataset)
    
 # ***************** NYTT ******************************


def noisy(image, noise_type):
    if noise_type is 0: #GAUSSIAN NOISE
        #row, col, ch = image.shape
        #mean = 0
        #var = 0.002
        #sigma = var**0.5
        #gauss = np.random.normal(mean,sigma,(row,col,ch))
        #gauss = gauss.reshape(row,col,ch)
        #noisy = image + gauss
        #return noisy


        row, col, ch = image.shape
        noisy = np.zeros((row,col,ch))
        half = int(ch/2)
  
        # Have lower noise on SUVr scans
        #SUVr
        mean = 0
        var = 0.0002
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row,col,half))
        gauss = gauss.reshape(row, col, half)
        noisy_suvr = image[:,:,0:half] + gauss

        # rCBF
        mean = 0
        var = 0.002
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row,col,half))
        gauss = gauss.reshape(row, col, half)
        noisy_rcbf = image[:,:,half:] + gauss

        noisy[:,:,0:half] = noisy_suvr
        noisy[:,:,half:] = noisy_rcbf

        return noisy


    elif noise_type is 1: # SALT AND PEPPER NOISE
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.2
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1*0.05

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0+0.0001

        return out

    elif noise_type is 2: # POSSION NOISE
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)

        return noisy

    elif noise_type is 3: # SPECKLE NOISE
        row,col,ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)        
        noisy = image + image * gauss
        
        return noisy

class noise2(object):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    def __init__(self, noise_type):
        self.noise_type = noise_type
        print("noise_type: ", noise_type)

    def __call__(self, img):
        """
        Args:
            img to be noised, assumes the images are between 0, 1

        Returns:
            img noised
        
        """

        img = noisy(img, self.noise_type)

        return img

class resize(object):
    """ Function to resize object to given size

    return resized object

    """
    def __call__(self, img,sizetobe=224):
        """
        Args:
            img to be noised, assumes the images are between 0, 1

        Returns:
            img noised
        
        """
        img = cv2.resize(img,dsize=(sizetobe,sizetobe))
        

        return img

class noise(object):
    def __call__(self, img):
        """
        Args:
            img to be noised, assumes the images are between 0, 1

        Returns:
            img noised
        
        """

        #img is a numpy array
        x,y,ch = img.shape

        #create random matrix of size img
        #noise = np.random.randn(x,y,ch) * 0.5 + 0.5 #Gives value between [0 1]
        noise = np.random.normal(loc = 0, scale = 0.1, size = (x,y,ch)) #Gives value between [0 1]

        img = img + noise
    
        return img

class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See filters_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        #rewrite from nmumpy to PIL
        angle = self.get_params(self.degrees)
        x,y,ch = img.shape
        for i in range(ch):
            img[:,:,i] = scp.rotate(img[:,:,i],angle,reshape=False)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string