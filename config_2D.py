import config

import torchvision
import torchvision.transforms as TT

#
# ***** create_dataset_2D.py *****
#
image_root = 'projectfiles_PE2I/scans/ecat_scans/'
label_root = 'projectfiles_PE2I/patientlist.csv'
filetype_SUVR = "1.v"
filetype_rCBF = "rCBF.v"
train_transform = TT.Compose([TT.RandomHorizontalFlip(p = 0.9),
                              TT.ToTensor(),
                              TT.RandomAffine(50, translate = None, scale = None, shear = None, resample=False, fillcolor=0)])


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
epochs = 10
batchSize = 5
learning_rate = 0.0001
optimizer = 'Adam'

#
# ***** model.py *****
#


