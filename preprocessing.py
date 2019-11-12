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

    print((len(dataset),sample[0].shape[0],sample[0].shape[1],sample[0].shape[2]))
    
    # Uses the reference images sizes
    SUVr = np.zeros((len(dataset),sample[0].shape[0],sample[0].shape[1],sample[0].shape[2]))
    rCBF = np.zeros((len(dataset),sample[1].shape[0],sample[1].shape[1],sample[1].shape[2]))

    for i in range(0,len(dataset)): # or i, image in enumerate(dataset)
        suvr_scan, rcbf_scan, label = dataset[i] # or whatever your dataset returns
        print(suvr_scan.shape)
        SUVr[i,:,:,:] = suvr_scan.squeeze()
        rCBF[i,:,:,:] = rcbf_scan.squeeze()


    return (SUVr.mean(), np.sqrt(SUVr.var()), rCBF.mean(), np.sqrt(rCBF.var()))


def find_centered_pixel(sample):
    image = np.squeeze(sample[0])

    val = np.max(image) - 0.15*np.max(image)  # HARD-CODED, USE PERCENTAGE INSTEAD? np.max(image) - 0.1*np.max(image)

    pixel_position = np.where(image > val)
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


if __name__ == "__main__":
    image_root = 'projectfiles_PE2I/scans/ecat_scans/'
    label_root = 'projectfiles_PE2I/patientlist.csv'
    filetype_SUVR = "1.v"
    filetype_rCBF = "rCBF.v"

    original_dataset = Get_dataset.ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF)
    sample = original_dataset.__getitem__(3)
    print(find_centered_pixel(sample))
    visFuncs.show_scan(sample,0)
    
    # *************** PREPROCESSING **********************
    nrPixelscrop_z = 40
    # Preprocessing stage: rotate crop, normalize
    preprocess = torchvision.transforms.Compose([
        #Get_dataset.Rotate_around_x(),
        Get_dataset.Crop(nrPixelscrop_z)
    ])

    # Calculate statistics of the rotated and cropped images to be able to normalize
    dataset = Get_dataset.ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF, preprocess)  
    sample = dataset.__getitem__(0)
    print(sample[0].shape[0])



    # Test the mean and variance calculations, REMEMBER EACH CHANNEL HAS ITS OWN MEAN AND STD
    SUVr_mean, SUVr_std, rCBF_mean, rCBF_std = calculate_mean_var(dataset) 
    print("SUVr_mean: ", SUVr_mean, "SUVr_std: ", SUVr_std, "rCBF_mean: ", rCBF_mean, "rCBF_std: ", rCBF_std) 

    # Normalize the dataset
    normalize = torchvision.transforms.Normalize((SUVr_mean, rCBF_mean), (SUVr_std, rCBF_std))
    dataset = normalize(dataset)

    #transform = Get_dataset.Crop(nrPixelscrop_z) #Argument is number of pixels to crop in Z direction, above and below
    #dataset_preprocessed = Get_dataset.ScanDataSet(image_root, label_root, filetype_SUVR, filetype_rCBF)

    #sample = dataset_preprocessed.__getitem__(0)
    # *************** PREPROCESSING **********************

    print(sample[0].shape)
    print()
    #visFuncs.show_scan(sample, 0)

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