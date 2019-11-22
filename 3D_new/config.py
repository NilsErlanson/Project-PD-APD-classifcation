import config

#
# ***** create_dataset.py *****
#
NumberOfAugmentations = 4
image_root = '../projectfiles_PE2I/scans/ecat_scans/'
label_root = '../projectfiles_PE2I/patientlist.csv'
filetype_SUVR = "1.v"
filetype_rCBF = "rCBF.v"

#
# ***** load_dataset.py *****
#
# Specified in the csv-file
nrOfDifferentDiseases = 6
# Cropping
fixed_degrees_z = 20
nrPixelsTop = 40
nrPixelsBottom = 40
# Rotation
random_degrees_x = 20
random_degrees_z = 5

#
# ***** training.py
#
epochs = 6
batchSize = 5

#
# ***** model.py *****
#

