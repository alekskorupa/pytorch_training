import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

label_names = ['bird',
               'boat',
               'cat',
               'dog',
               'flower',
               'frog',
               'jumbojet',
               'mushroom',
               'sportscar',
               'tree']


def plot_images(imgs, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    plt.figure(figsize=(8,8))
    # show images
    imshow(make_grid(imgs))
    print(cls_true)

    plt.show()
