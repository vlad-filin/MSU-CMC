import numpy as np
from importlib import reload
import mixture_model as mm
import time as t
import matplotlib.pyplot as plt
import imageio
from skimage.io import imshow
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.colors import rgb_to_hsv

def load_data(path, train=True):
    if train is True:
        data = np.zeros((299, 240, 360, 3), dtype=np.int)
        for i in range(0, 299):
            name = "/in" + req_num(i+1) + ".jpg"
            data[i,...] = imageio.imread(path + name)
    else: 
        data = np.zeros((800, 240, 360, 3), dtype=np.int)
        for i in range(299, 1099):
            name = "/in" + req_num(i+1) + ".jpg"
            data[i-299,...] = imageio.imread(path + name)
    return data
def load_true(path="../29.03/pedestrians/groundtruth"):
    ans = np.zeros((800, 240, 360), dtype=np.bool)
    for i in range(299, 1099):
            name = "/gt" + req_num(i+1) + ".png"
            ans[i-299,...] = imageio.imread(path + name)
    return ans
            
def req_num(i):
    t = len(str(i))
    return "0" * (6-t) + str(i)
def convert_data_to_gray(x):
    """
    x = [n_frames,H,W,channels]
    out = [n_frames, H, W]
    """
    f, h, w, _ = x.shape
    out = np.zeros((f, h, w), dtype=np.int)
    for i in range(f):
        out[i,...] = (0.2126 * x[i,:,:,0] + 0.7152 * x[i,:,:,1] + 0.0722 * x[i,:,:,2]).astype(int)
    return out
def convert_data_to_hsv(x):
    f, h, w, c = x.shape
    output = np.zeros((f, h, w, c), dtype=int)
    for i in range(f):
        output[i,...] = (rgb_to_hsv(x[i,...] / 255)*100).astype(int)
    return output
def make_big_image(l_x, size=(2,4)):
    f, h, w, c = l_x.shape
    output = np.zeros(((h + 3) * size[0], (w+3)*size[1], c), dtype=np.uint8)
    sh_h, sh_w = 0, 0
    for i in range(size[0]):
        sh_w=0
        for j in range(size[1]):
            output[i * h + sh_h: (i+1) * h + sh_h, j * w + sh_w: (j+1) * w + sh_w, :] = l_x[i + j, ...]
            sh_w += 3
        sh_h += 3
    return output
def count_FP_FN(true_ans, pred_ans):
    F, _, _, = true_ans.shape
    FP = []
    FN = []
    for f in range(F):
        t = true_ans[f,...].ravel()
        p = pred_ans[f,...].ravel()
       # TP = ((t == True) & (p==True)).sum()
        FP += [((t == False) & (p==True)).sum()]
        FN += [((t == True) & (p==False)).sum()]
       # TN = ((t == False) & (p==False)).sum()
    return FP, FN