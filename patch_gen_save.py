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
save_fdir="/mnt/nfs/kyuhyoung/pathology/breast/hyosari/Tummor_patches/"
save_pdir="/mnt/nfs/kyuhyoung/pathology/breast/hyosari/Tummor_patches/Selection_Point/"
save_cdir="/mnt/nfs/kyuhyoung/pathology/breast/hyosari/Tummor_patches/Selection_Point/"


slide_level = 4
patch_size = 256
patch_num=1
patch_generation(tif_fdir,jpg_fdir,save_fdir,save_pdir,save_cdir,slide_level,patch_size,patch_num)




"""
ilist_tif_name = [ f for f in listdir(tif_fdir)] 
list_tif_name.sort()

list_jpg_name = [ f for f in listdir(jpg_fdir)]
list_jpg_name.sort()


for i in range(1):
    pwd_tif = join(tif_fdir,list_tif_name[i])
    pwd_jpg = join(jpg_fdir,list_jpg_name[i])
    pwd_save = splitext(list_tif_name[i])[0] 
    pwd_point = splitext(list_tif_name[i])[0]
    pwd_csv = splitext(list_tif_name[i])[0]

    slide = OpenSlide(pwd_tif) 
    f= open(save_cdir+pwd_csv+".csv",'wt')
    c_writer = csv.writer(f)
    c_writer.writerow( ('Y', 'X') )
    slide_level=4
    rgba_pil= slide.read_region((0,0),slide_level,slide.level_dimensions[slide_level])
    rgb_im = cv2.cvtColor(np.array(rgba_pil),cv2.COLOR_RGBA2BGR) 
    mask = cv2.imread(pwd_jpg,cv2.IMREAD_GRAYSCALE) 
    patch_size = 256 
    patch_num = 3

    if np.max(mask) == 255:
        mask = mask == 255
        mask = mask.astype(int) 

    patch_list, patch_point = patch_sampling_using_integral(slide,slide_level,mask,patch_size,patch_num)
    p_l_size =patch_size/slide.level_downsamples[slide_level]
    p_l_size = int(p_l_size) 

    for f_th in range(patch_num): 
        cv2.imwrite(save_fdir+pwd_save+"_patch_"+str(f_th)+"_"+str(patch_point[f_th][1])+"_"+str(patch_point[f_th][0])+"_"+str(patch_size)+".jpg",patch_list[f_th])
        
        #print pwd_save+"_patch_"+str(f_th)+".jpg is written" 
        c_writer.writerow( (patch_point[f_th][1], patch_point[f_th][0]))
        cv2.rectangle(rgb_im,(patch_point[f_th][1],patch_point[f_th][0]),(patch_point[f_th][1]+p_l_size,patch_point[f_th][0]+p_l_size),(255,0,0),3)

    
    cv2.imwrite(save_pdir+pwd_save+"_selection_point.jpg",rgb_im) 
    print "complete patch extraction about "+list_tif_name[i]
    f.close()
#   plt.figure("selection")   
#   plt.imshow(rgb_im)
#   plt.show()
"""


















