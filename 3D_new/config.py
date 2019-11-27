import config

#
# ***** create_dataset.py *****
#
image_root = '../projectfiles_PE2I/scans/ecat_scans/'
label_root = '../projectfiles_PE2I/patientlist.csv'
filetype_SUVR = "1.v"
filetype_rCBF = "rCBF.v"
applyGammaTransformation = 1

#
# ***** load_dataset.py *****
#
# Specified in the csv-file
nrOfDifferentDiseases = 4
# Cropping
fixed_degrees_z = 20
nrPixelsTop = 40
nrPixelsBottom = 40
cropx = 100
cropy = 100
# Rotation
random_degrees_x = 20
random_degrees_z = 5

#
# ***** training.py
#
USE_CUDA = True
epochs = 10
batchSize = 4
learning_rate = 0.0001

#
# ***** model.py *****
#

