import config

import torchvision
import torchvision.transforms as TT

import transformations_2D
#
# ***** create_dataset_2D.py *****
#
image_root = '../projectfiles_PE2I/scans/ecat_scans/'
label_root = '../projectfiles_PE2I/patientlist.csv'
filetype_SUVR = "1.v"
filetype_rCBF = "rCBF.v"

sliceSample = 64 # If one wants the whole dataset, put None
addMeanImage = True # or True
numberOfmeanslices = 5

#train_transform = TT.Compose([#TT.ToPILImage(),
                              #TT.RandomHorizontalFlip(p = 0.9)])
                              #transformations_2D.noise2(0),
                              #transformations_2D.RandomRotation(20),
 #                             TT.ToTensor()])

#test_transform = TT.Compose([TT.ToTensor()])

# resent 18
train_transform = TT.Compose([
    #TT.ToPILImage(),
    transformations_2D.resize(),
    #TT.CenterCrop(224),
    TT.ToTensor(),
<<<<<<< HEAD
   # TT.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25, 0.25]),
=======
    #TT.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25, 0.25]),
>>>>>>> andreas
])

test_transform = TT.Compose([
    #TT.ToPILImage(),
    #TT.Resize(256),
    #TT.CenterCrop(224),
    transformations_2D.resize(),
    TT.ToTensor(),
<<<<<<< HEAD
   # TT.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25, 0.25]),
=======
    #TT.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25, 0.25]),
>>>>>>> andreas
])



#
# ***** load_dataset.py *****
#
# Specified in the csv-file
nrOfDifferentDiseases = 4

#
# ***** training.py
#
USE_CUDA = True # NO = 0, YES = 1
<<<<<<< HEAD
epochs = 20
=======
epochs = 5
>>>>>>> andreas
batchSize = 4
learning_rate = 0.0001
optimizer = 'Adam'

#
# ***** model.py *****
#
if addMeanImage == False:
    inChannels = 2
else:
    inChannels = 4




