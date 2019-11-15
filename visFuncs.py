import matplotlib.pyplot as plt
import numpy as np


def scroll_slices(sample):
    images = np.squeeze(sample[0])

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
    
    if(np.array_equal(label,[1, 0, 0, 0, 0, 0])):
        name = 'Frisk'
    elif(np.array_equal(label,[0, 1, 0, 0, 0, 0])):
        name = 'Vaskulär Parkinomism'
    elif(np.array_equal(label,[0, 0, 1, 0, 0, 0])):
        name = 'PD'
    elif(np.array_equal(label,[0, 0, 0, 1, 0, 0])):
        name = 'Lewy Body Disease'
    elif(np.array_equal(label,[0, 0, 0, 0, 1, 0])):
        name = 'FTD'
    else:
        name = 'NPH'
    return name


def show_scan(sample,sliceNr=10,show = True):
    images = sample[0]
    label = sample[1]
    name = get_name(label)
    img_suvr = images[...,0]
    img_rbf = images[...,1]
    fig, axs = plt.subplots(2, 3, figsize=(10,10))


    #SUVR first row
    axs[0,0].imshow(img_suvr[:,:,sliceNr],cmap='bone') #SUVr
    axs[0,0].set_title(['SUVR, Slicenumber: ', sliceNr])
    axs[0,1].imshow(img_suvr[:,:,sliceNr+40],cmap = 'bone') #SUVr
    axs[0,1].set_title(['suvr, slicenumber: ', sliceNr+40])
    axs[0,2].imshow(img_suvr[60,:,:],cmap = 'bone')
    axs[0,2].set_title('rotated')
    #RBF secod row
    axs[1, 0].imshow(img_rbf[:,:,sliceNr],cmap = 'bone') #rCBF
    axs[1, 0].set_title(['rcbf, slicenumber: ', sliceNr])
    axs[1, 1].imshow(img_rbf[:,:,sliceNr+39],cmap = 'bone') #rCBF
    axs[1, 1].set_title(['rcbf, slicenumber ', sliceNr+40])
    axs[1, 2].imshow(img_rbf[60,:,:],cmap = 'bone') #rCBF
    axs[1, 2].set_title(['rcbf, rotated '])
    
    plt.suptitle(name)
    if(show):
        plt.show()

def comp_samples(sample1,sample2):

    show_scan(sample1,show = False)

    show_scan(sample2,show = False)
    plt.show()