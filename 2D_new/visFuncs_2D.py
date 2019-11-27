import matplotlib.pyplot as plt
import numpy as np


def scroll_slices(sample, original = False):
    images = sample[0]
    if original is False:
        images = images.permute(1,2,0)

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

    tracker = IndexTracker(ax,images)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
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