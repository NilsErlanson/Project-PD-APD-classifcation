import config

import torchvision
import torchvision.transforms as TT

import transformations_2D
#
# ***** create_dataset_2D.py *****
#
image_root = 'projectfiles_PE2I/scans/ecat_scans/'
label_root = 'projectfiles_PE2I/patientlist.csv'
filetype_SUVR = "1.v"
filetype_rCBF = "rCBF.v"
train_transform = TT.Compose([#TT.ToPILImage(),
                              #TT.RandomHorizontalFlip(p = 0.9)])
                              transformations_2D.noise2(3),
                              transformations_2D.RandomRotation(20),
                              TT.ToTensor()])


test_transform = TT.Compose([TT.ToTensor()])

#
# ***** load_dataset.py *****
#
# Specified in the csv-file
nrOfDifferentDiseases = 5
# Cropping
fixed_degrees_z = 20
nrPixelsTop = 30
nrPixelsBottom = 50
# Rotation
random_degrees_x = 20
random_degrees_z = 5

#
# ***** training.py
#
USE_CUDA = True # NO = 0, YES = 1
epochs = 1
batchSize = 5
learning_rate = 0.0001
optimizer = 'Adam'

#
# ***** model.py *****
#


