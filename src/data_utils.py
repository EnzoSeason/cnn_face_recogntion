import numpy as np
from skimage import transform

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

def generate_neg_data(img_raw, label, idx_start, idx_end, h_fixed, l_fixed, ratio_HL):
    data_neg = []
    
    for i in range(len(img_raw[idx_start:idx_end])):
        H = img_raw[i].shape[0]
        L = img_raw[i].shape[1]
        image = img_raw[i]
        exemples_pos = []
        for face in label:
            if face[0] == i+1:
                exemples_pos.append(face)
        nb_img_neg = 0
        while nb_img_neg < 5:
            position_h = int(np.random.uniform(0,H))
            position_l = int(np.random.uniform(0,L))
            # h and l need to be fine-tined
            h = int(np.random.uniform(int(H/5),int(H/2))) 
            l = int(h/ratio_HL)
            img_atr = np.array([position_h, position_l, h, l])
            is_pos = False
            for face in exemples_pos:
                aire_re = calcul_aire_recouvrement(face[1:5], img_atr)
                if aire_re > 0.1:
                    is_pos = True
            if is_pos == False:
                img_neg = image[position_h: position_h + h, position_l: position_l + l]
                img_neg = transform.resize(img_neg, (h_fixed, l_fixed))
                data_neg.append(img_neg)
                nb_img_neg += 1
    
    data_neg = np.array(data_neg)
    return data_neg