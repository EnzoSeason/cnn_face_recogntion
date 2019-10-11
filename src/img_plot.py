import sys
import numpy as np

import matplotlib.pyplot as plt 
from skimage import io, util, color, transform

import random

def plot_images_in_square(img_array, n, plot_size):
    fig, ax = plt.subplots(n, n, figsize=(plot_size, plot_size))
    idx = 0
    nb_sample = img_array.shape[0]
    img_idx = random.sample(range(nb_sample), n*n)
    for i in range(n):
        for j in range(n):
            ax[i,j].imshow(img_array[img_idx[idx],:,:])
            idx += 1
    plt.show()