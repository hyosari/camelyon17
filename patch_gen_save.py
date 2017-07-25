from openslide import OpenSlide
from os import listdir
from os.path import join, isfile, exists, splitext
import cv2 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
import csv

from image_plot import subplot_show 
from integral import patch_sampling_using_integral 
from patch_generation import patch_generation

# prepare files list 

tif_fdir ="/mnt/nfs/kyuhyoung/pathology/breast/camelyon16/TrainingData/Train_Tumor/"
jpg_fdir ="/mnt/nfs/kyuhyoung/pathology/breast/bong/Slide_Ground_Truth_lv_4/"
save_fdir="/mnt/nfs/kyuhyoung/pathology/breast/hyosari/camelyon16/tumor_patches"
save_pdir="/mnt/nfs/kyuhyoung/pathology/breast/hyosari/camelyon16/tumor_patches/selection_points"
save_cdir="/mnt/nfs/kyuhyoung/pathology/breast/hyosari/camelyon16/tumor_patches/selection_points"


slide_level = 4
patch_size = 224
patch_num=1000
patch_generation(tif_fdir,jpg_fdir,save_fdir,save_pdir,save_cdir,slide_level,patch_size,patch_num,tumor_win=True)

        
"""
#   plt.figure("selection")   
#   plt.imshow(rgb_im)
#   plt.show()
"""


















