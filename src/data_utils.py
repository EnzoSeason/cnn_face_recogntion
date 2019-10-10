import numpy as np

def calcul_aire_recouvrement(line1, line2):
    H = int(max(line1[0]+line1[2], line2[0]+line2[2]))
    L = int(max(line1[1]+line1[3], line2[1]+line2[3]))
    bg1 = np.zeros((H, L))
    bg2 = np.zeros((H,L))
    for i in np.arange(int(line1[0]),int(line1[0]+line1[2])):
        for j in np.arange(int(line1[1]), int(line1[1]+line1[3])):
            bg1[i,j] = 1
    for i in np.arange(int(line2[0]),int(line2[0]+line2[2])):
        for j in np.arange(int(line2[1]), int(line2[1]+line2[3])):
            bg2[i,j] = 1        
    uni = np.logical_or(bg1,bg2).sum()
    inter = np.logical_and(bg1, bg2).sum()
    return inter/uni