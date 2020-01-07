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

# For the transformations
sliceSample = 67
applydiffnormal = True
useMultipleSlices = False
applyMirrorImage = True
cropx = 110
cropy = 100
gamma = 1.025

transform_train = TT.Compose([
    transformations_2D.resize(),
    TT.ToTensor()
])

transform_test = TT.Compose([
    transformations_2D.resize(),
    TT.ToTensor()
])

#
# ***** load_dataset.py *****
#
# Specified in the csv-file
nrOfDifferentDiseases = 4

#
# ***** training.py
#
USE_CUDA = True
epochs = 20
batchSize = 16
learning_rate = 9.0e-6
nrAugmentations = 8

#
# ***** model.py *****
#
if applydiffnormal == False:
    inChannels = 2
else:
    inChannels = 4