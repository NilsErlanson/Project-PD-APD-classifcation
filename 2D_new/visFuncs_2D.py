import matplotlib.pyplot as plt
import numpy as np

def plot_k_fold(train_losses, test_losses):
    mean_train = np.mean(train_losses, axis=0)
    mean_test = np.mean(test_losses, axis=0)

    # plot loss
    plt.figure()
    iterations = np.arange(1, len(mean_train) + 1)
    plt.scatter(iterations, mean_train, label = 'mean training loss')
    plt.scatter(iterations, mean_test, label = 'mean test loss')
    plt.legend()
    plt.xlabel('iteration')
    plt.show()   

def plot_training_test_loss(training_loss, test_loss):
    # plot loss
    plt.figure()
    iterations = np.arange(1, len(training_loss) + 1)
    plt.scatter(iterations, training_loss, label='training loss')
    plt.scatter(iterations, test_loss, label='test loss')
    plt.legend()
    plt.xlabel('iteration')
    plt.show()  

def get_name(label):
    
    if(np.array_equal(label,[1, 0, 0, 0])):
        name = 'Frisk'
    elif(np.array_equal(label,[0, 1, 0, 0])):
        name = 'Vaskulär Parkinomism'
    elif(np.array_equal(label,[0, 0, 1, 0])):
        name = 'PD'
    else:
        name = 'Lewy Body Disease'

    return name

def show_scan_multipleSlices(sample, show = True):
    images = sample[0]
    label = sample[1]
    name = get_name(label)

    fig, axs = plt.subplots(2, 1, figsize=(10,10))

    #SUVR
    axs[0].imshow(images[0,:,:], cmap='hot')
    axs[0].set_title(['SUVR'])
    #RCBF
    axs[1].imshow(images[1,:,:], cmap = 'hot') 
    axs[1].set_title(['RCBF'])

    plt.suptitle(name)
    if(show):
        plt.show()

def show_scan(sample, original=False, show = True):
    sliceNr_SUVr = 40
    sliceNr_rCBF = 120
    images = sample[0]

    if original is False:
        images = images.permute(1,2,0)

    label = sample[1]
    name = get_name(label)

    fig, axs = plt.subplots(2, 3, figsize=(10,10))


    #SUVR first row
    axs[0,0].imshow(images[:,:,sliceNr_SUVr], cmap='bone') #SUVr
    axs[0,0].set_title(['SUVR, Slicenumber: ', sliceNr_SUVr])
    axs[0,1].imshow(images[:,:,sliceNr_SUVr+20], cmap = 'bone') #SUVr
    axs[0,1].set_title(['suvr, slicenumber: ', sliceNr_SUVr+20])
    axs[0,2].imshow(images[60,:,:], cmap = 'bone')
    axs[0,2].set_title('rotated')

    #RBF secod row
    axs[1, 0].imshow(images[:,:,sliceNr_rCBF], cmap = 'bone') #rCBF
    axs[1, 0].set_title(['rcbf, slicenumber: ', sliceNr_rCBF])
    axs[1, 1].imshow(images[:,:,sliceNr_rCBF+20], cmap = 'bone') #rCBF
    axs[1, 1].set_title(['rcbf, slicenumber ', sliceNr_rCBF+20])
    axs[1, 2].imshow(images[60,:,:], cmap = 'bone') #rCBF
    axs[1, 2].set_title(['rcbf, rotated '])
    
    plt.suptitle(name)
    if(show):
        plt.show()

def comp_samples(sample1,sample2):

    show_scan(sample1,show = False)

    show_scan(sample2,show = False)
    plt.show()