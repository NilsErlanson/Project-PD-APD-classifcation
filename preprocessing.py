import os
import nibabel as nib
nibabel_dir = os.path.dirname(nib.__file__)
from nibabel import ecat
import pandas as pd
import numpy as np
import sklearn

#Our own files
import Get_dataset
import visFuncs

# To rotate the images
import scipy.ndimage

#To transform the images
import torchvision

import matplotlib.pyplot as plt

import torchvision

def scroll_slices(image):

    fig, ax = plt.subplots(1, 1)
    class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('use scroll wheel to navigate images')

            self.X = X
            rows, cols, self.slices = X.shape
            self.ind = self.slices//2

            self.im = ax.imshow(self.X[:, :, self.ind])
            self.update()

        def onscroll(self, event):
            print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def update(self):
            self.im.set_data(self.X[:, :, self.ind])
            ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()

    tracker = IndexTracker(ax,image)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

def calculate_mean_var(dataset):
    #Fetch a reference sample to be able to create matrices of the right size
    sample = dataset.__getitem__(0)

    image = sample[0][:,:,:,0]
    plt.imshow(image[64,:,:]) 
    plt.show()

    shape = sample[0].shape
    print((len(dataset),shape[0], shape[1], shape[2]))
    
    # Uses the reference images sizes
    images = np.zeros((len(dataset), shape[0], shape[1], shape[2], 2))
    print(images.shape)

    for i in range(0, len(dataset)): # or i, image in enumerate(dataset)
        sample = dataset.__getitem__(i) # or whatever your dataset returns
        print("In loop SUVr.shape (SCANS): ", sample[0][:,:,:,0].shape)

        images[i,:,:,:,0] = sample[0][:,:,:,0] #SUVr
        images[i,:,:,:,1] = sample[0][:,:,:,1] #rCBF

    return (images[:,:,:,:,0].mean(), np.sqrt(images[:,:,:,:,0].var()), images[:,:,:,:,1].mean(), np.sqrt(images[:,:,:,:,1].var()))


def find_centered_pixel(sample):
    #image = np.squeeze(sample[0])
    image = sample[0][:,:,:,0] #SUVr

    val = np.max(image) - 0.15*np.max(image)  # HARD-CODED, USE PERCENTAGE INSTEAD? np.max(image) - 0.1*np.max(image)
    pixel_position = np.where(image > val)

    #print("val: ", val, "pixel_pos: ", pixel_position)

    #val = np.max(sample[0][:,:,0:100,0]) - np.max(sample[0][:,:,0:100,0]) * 0.35 # SUVR IMAGE
    #pixel_position  = np.where( sample[0][:,:,0:100,1] > val )

    x = pixel_position[0][:]
    y = pixel_position[1][:]
    z = pixel_position[2][:]
    #Get middle position of the two balls, locate x-position
    threshold = (np.max(x) + np.min(x)) / 2
    index = x <= threshold
    xmid = (np.mean(x[index])+np.mean(x[~index]))/2
    ymid = np.mean(y).astype(int)
    zmid = np.mean(z).astype(int)

    #Point between balls
    position = np.array([xmid,ymid,zmid]).astype(int)
    #print(position)

    return position

def crop_image(sample, position, nrPixelsxy, nrPixelsz):
    image = np.squeeze(sample[0])
    
    # Crop xy as well
    #cropped_image  = image[position[0]- nrPixelsxy:position[0]+nrPixelsxy, position[1]-nrPixelsxy:position[1]+nrPixelsxy, position[2]-nrPixelsz:position[2]+nrPixelsz]

    #Only crop in z
    cropped_image  = image[:,:, position[2]-nrPixelsz:position[2]+nrPixelsz]

    #print(cropped_image.shape)
    return cropped_image

#
# Data augmentation functions
#

def shrink_image(sample):

    return shrinked_image

def enlarge_image(sample):

    return enlarged_image

def rotate_image(sample):

    return rotated_sample

def noise_dataset(orginial_dataset, transform):


    return noise_dataset

if __name__ == "__main__":
    image_root = 'projectfiles_PE2I/scans/ecat_scans/'
    label_root = 'projectfiles_PE2I/patientlist.csv'
    filetype_SUVR = "1.v"
    filetype_rCBF = "rCBF.v"

    original_dataset = Get_dataset.ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF)
    sample = original_dataset.__getitem__(0)
    #print(find_centered_pixel(sample))
    #visFuncs.show_scan(sample,0)
    
    # *************** PREPROCESSING **********************
    pixelstop = 30
    pixelsbottom = 50
    degrees = 20
    # Preprocessing stage: rotate crop, normalize
    preprocess = torchvision.transforms.Compose([
        Get_dataset.Rotate_around_x(degrees),
        Get_dataset.Crop(pixelstop, pixelsbottom)
    ])

    #Only rotation
    #preprocess = Get_dataset.Rotate_around_x(degrees)
    #Only cropping
    preprocess = Get_dataset.Crop(pixelstop, pixelsbottom)
    
    #With preprocess
    #dataset = Get_dataset.ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF, preprocess)
    #Without preprocess  
    dataset1 = Get_dataset.ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF)  
    #sample = dataset.__getitem__(0)
    dataset2 = Get_dataset.ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF)  
    #print("sample[0].shape: ", sample[0].shape, "sample[1].shape: ", sample[1].shape)


    # Calculate statistics of the rotated and cropped images to be able to normalize
    #dataset = Get_dataset.ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF, preprocess)  
    #sample = dataset.__getitem__(0)
    #print(sample[0].shape[0])

    #Show the image
    #image = sample[0][:,:,:,0]
    #plt.imshow(image[64,:,:]) 
    #plt.show()

    # Test the mean and variance calculations, REMEMBER EACH CHANNEL HAS ITS OWN MEAN AND STD
    #SUVr_mean, SUVr_std, rCBF_mean, rCBF_std = calculate_mean_var(dataset) 
    #print("SUVr_mean: ", SUVr_mean, "SUVr_std: ", SUVr_std, "rCBF_mean: ", rCBF_mean, "rCBF_std: ", rCBF_std) 

    # Normalize the dataset
    #normalize = torchvision.transforms.Normalize((SUVr_mean, rCBF_mean), (SUVr_std, rCBF_std))
    #dataset = normalize(dataset)

    #transform = Get_dataset.Crop(nrPixelscrop_z) #Argument is number of pixels to crop in Z direction, above and below
    #dataset_preprocessed = Get_dataset.ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF)

    #sample = dataset_preprocessed.__getitem__(0)
    # *************** PREPROCESSING **********************

    print(sample[0].shape)
    print()
    #visFuncs.show_scan(sample, 0)

    """
    bild=9

    position = find_centered_pixel(sample)

    #Crop according to position of striatum middle point(ignoring cropping in x & y)
    nrPixelsz = 30 #top
    z2 = 50 #bottom
    SUV_cropped = imageSUV_rot[:,:, position[2]-z2 : position[2]+nrPixelsz]
    CBF_cropped = imageCBF_rot[:,:, position[2]-z2 : position[2]+nrPixelsz]

    fig, axs = plt.subplots(3, 2, figsize=(10,10))
    sliceNr1=64
    sliceNr2=64

    axs[0, 0].imshow(imageSUV[64,:,:],cmap='hot') #SUVr 
    axs[0, 0].set_title(['Patient 1: SUVr (PET) slice ', sliceNr1])

    axs[0, 1].imshow(imageCBF[64,:,:],cmap='hot') #SUVr
    axs[0, 1].set_title(['Patient 1: rCBF (SPECT) slice ', sliceNr2])

    axs[1,1].scatter(position[2],position[1],position[0])
    axs[1, 0].imshow(imageSUV_rot[64,:,:],cmap='hot') #SUVr
    axs[1, 0].set_title(['Patient 1: rCBF (SPECT) slice ', sliceNr2])

    axs[1, 1].imshow(imageCBF_rot[64,:,:],cmap='hot') #SUVr
    axs[1, 1].set_title(['Patient 1: rCBF (SPECT) slice ', sliceNr2])
    axs[2, 0].imshow(SUV_cropped[64,:,:],cmap='hot') #SUVr
    axs[2, 0].set_title(['Patient 1: rCBF (SPECT) slice ', sliceNr2])
    axs[2, 1].imshow(CBF_cropped[64,:,:],cmap='hot') #SUVr
    axs[2, 1].set_title(['Patient 1: rCBF (SPECT) slice ', sliceNr2])
    """


    """
    #Find the centered pixel position according to maximum intensity
    centered_pixel_position = find_centered_pixel(sample)

    # Find the "centered pixel position" based on the two maximum intensity areas in the image
    centered_pixel_pos = find_centered_pixel(sample)
    # Manually perform some crop
    xycrop = 40
    zcrop = 45
    # Returns a cropped image with ***zcrop*** as the height with the pixel that has the maximum pixelvalue 
    cropped_image = crop_image(sample, centered_pixel_pos, xycrop, zcrop)

    #plott the resulting cropped image
    #plt.imshow(cropped_image[75,:,:])
    #plt.show()




    # Define a transform, based on mean and var statistics attained from the dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((mean), (std))])   # REMEMBER EACH CHANNEL HAS ITS OWN MEAN AND STD

    # TEST OUR OWN TRANSFORMATIONS
    nrOfPixelsz = 40
    crop_transformation = Get_dataset.Crop(nrOfPixelsz)

    cropped_sample = crop_transformation(sample)
    print(cropped_sample[0].shape)


    #dataset_normalized = torchvision.transforms.Normalize(dataset,(mean), (std))

    # A test to implement a dataloader to perfrom transformations to the dataset
    #dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
    #                                         batch_size=3, shuffle=True,
    #                                         num_workers=1)

    #Compare the normalized max value to the standard image
    #sample_normalized = dataset_normalized.__getitem__(0)
    #print("Before normalization ", np.max(np.squeeze(sample[0])))
    #print("Before normalization ", np.max(np.squeeze(sample_normalized[0])))

    dataset_normalized = Get_dataset.ScanDataSet_transform(image_root,label_root,filetype)  

    
    # Manually perform some crop
    #nrPixels = 45
    #cropped  = image[position[0][0]-nrPixels:position[0][0]+nrPixels, position[1][0]-nrPixels:position[1][0]+nrPixels, position[2][0]-nrPixels:position[2][0]+nrPixels]

    #plt.imshow(cropped) #SUVr
    #plt.scatter(position[1][0], position[0][0])
    #plt.show()

    #print(cropped.shape)

    #scroll_slices(cropped)

    # Save the file as DICOM
    #write_dicom(cropped,'cropeed.dcm')
    #vtkarray = numpy_support.numpy_to_vtk(cropped)
    """