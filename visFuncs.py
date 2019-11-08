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

def show_scan(sample,sliceNr=10):
    images = np.squeeze(sample[0])
    label = sample[1]
    
    fig, axs = plt.subplots(2, 2, figsize=(10,10))

    axs[0, 0].imshow(images[:,:,sliceNr]) #SUVr
    axs[0, 0].set_title(['Slicenumber', sliceNr])
    axs[0, 1].imshow(images[:,:,sliceNr+30]) #rCBF
    axs[0, 1].set_title(['Slicenumber', sliceNr+30])
    axs[1, 0].imshow(images[:,:,sliceNr+60]) #SUVr
    axs[1, 0].set_title(['slicenumber ', sliceNr+60])
    axs[1, 1].imshow(images[:,:,sliceNr+90]) #rCBF
    axs[1, 1].set_title(['slicenumber ', sliceNr+90])
    plt.show()