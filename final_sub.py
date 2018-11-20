import numpy as np
import pandas as pd
from skimage.morphology import label

def rle_to_mask(rle_list, SHAPE):
    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i,v in zip(strt,length):
            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 1.0
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

  # ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
s1 = pd.read_csv('../input/airbus-dataset/sub.csv')
s2 = pd.read_csv('../input/airbus-dataset/sub_1.csv')
s = pd.concat([s1, s2], axis=0)
s_gp = s.groupby('ImageId').sum()
s_gp = s_gp.reset_index()
pred_rows = []
for image_name in s_gp['ImageId'].tolist():
    mask_list = s['EncodedPixels'][s['ImageId'] == image_name].tolist()
    seg_mask = np.zeros((768, 768))
    for item in mask_list:
        rle_list = str(item).split()
        tmp_mask = rle_to_mask(rle_list, (768, 768))
        seg_mask[:,:] += tmp_mask
    seg_mask = np.where(seg_mask<=1, 0, 1)
    rles = multi_rle_encode(seg_mask)
    if len(rles)>0:
        for rle in rles:
            pred_rows += [{'ImageId': image_name, 'EncodedPixels': rle}]
    else:
        pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]
submission_df = pd.DataFrame(pred_rows)[['ImageId', 'EncodedPixels']]
submission_df.to_csv('submission.csv', index=False)
