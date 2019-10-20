import sys
import numpy as np
from itertools import cycle

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

def plot_roc(fpr, tpr, roc_auc, type, class_index=None):
    '''
    Parameters:
    fpr, tpr, roc_auc must be the dict.
    '''
    plt.figure()
    lw = 2
    n_classes = len(roc_auc) - 2
    if type == 'each_class':
        if class_index == None:
            print('No class given')
            return None
        plt.plot(fpr[class_index], tpr[class_index], color='darkorange',
                lw=lw, label='ROC curve of class {0} (area = %0.2f)'.format(class_index, roc_auc[class_index]))
    
    if type == 'all_classes' or type == 'all':
        colors = cycle(['aqua', 'darkorange'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    if type == 'micro' or type == 'all':
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

    if type == 'macro' or type == 'all':
        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()