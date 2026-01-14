import numpy as np
import glob 
import os
from PIL import Image
import math
import SimpleITK as sitk
import cv2
import random
import nibabel as nb
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim


# function: set window level
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width // 2
    low = level - width // 2
    # normalize
    unit = (1-0) / (width)
    new[new>high] = high
    new[new<low] = low
    new = (new - low) * unit 
    return new

# function: get first X numbers
# if we have 1000 numbers, how to get the X number of every interval numbers?
def get_X_numbers_in_interval(total_number, start_number, end_number , interval = 100):
    '''if no random pick, then random_pick = [False,0]; else, random_pick = [True, X]'''
    n = []
    for i in range(0, total_number, interval):
        n += [i + a for a in range(start_number,end_number)]
    n = np.asarray(n)
    return n


# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: find time frame of a file
def find_timeframe(file,num_of_dots,start_signal = '/',end_signal = '.'):
    k = list(file)

    if num_of_dots == 0: 
        num = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk = k[num+1:]
    
    else:
        if num_of_dots == 1: #.png
            num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        elif num_of_dots == 2: #.nii.gz
            num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
        num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk=k[num2+1:num1]


    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    
    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)


# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True, WL = 50, WW = 100):
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = set_window(a, WL, WW)

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path)

# function: plot ROC curve and get AUC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def plot_roc_curve(y_true, y_prob, figsize=(6,6)):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0,1], [0,1], '--', lw=1)   # random guess line
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Out-of-fold ROC curve (5-fold CV)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# function: calculate ICC(2,1)
def icc2_1(x, y):
    """
    ICC(2,1): Two-way random effects, absolute agreement, single measurement.
    x, y: 1D arrays, same length (subjects)
    returns: ICC value (float)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # remove NaNs pairwise
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 2:
        return np.nan

    # data matrix: n subjects x k raters (k=2)
    Y = np.column_stack([x, y])
    n, k = Y.shape  # k=2

    mean_row = Y.mean(axis=1, keepdims=True)   # subject means
    mean_col = Y.mean(axis=0, keepdims=True)   # rater means
    grand_mean = Y.mean()

    # Sum of squares
    SSR = k * np.sum((mean_row - grand_mean) ** 2)          # rows (subjects)
    SSC = n * np.sum((mean_col - grand_mean) ** 2)          # columns (raters)
    SSE = np.sum((Y - mean_row - mean_col + grand_mean) ** 2)  # residual

    # Mean squares
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    # ICC(2,1)
    denom = MSR + (k - 1) * MSE + (k * (MSC - MSE) / n)
    if denom == 0:
        return np.nan
    icc = (MSR - MSE) / denom
    return icc

