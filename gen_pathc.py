from openslide import OpenSlide
from os import listdir
from os.path import join, isfile, exists, splitext
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from util import otsu_thresholding

pwd ="/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/centre_4/"
filename = "patient_099_node_4.tif" 

slide_path = join(pwd,filename) 

#Open tif file !! 
slide = OpenSlide(slide_path) 

print "WSI Dimensions (width , hight) : {0}".format(slide.dimensions) 

#Slide.read reagion 
slide_level=3
rgba_im = slide.read_region((0,0),slide_level,slide.level_dimensions[slide_level])
rgba_im= np.array(rgba_im)

rgb_im = cv2.cvtColor(rgba_im,cv2.COLOR_RGBA2RGB) 

#convert to gray 
gray_im = cv2.cvtColor(rgb_im,cv2.COLOR_RGB2GRAY) 
#apply atsu thresholding 
otsu_im,o_th = otsu_thresholding(gray_im)
#Draw Contours
cont_im = rgb_im 
#cv2.drawContours(cont_im,otsu_im,-1,(0,255,0),-1) 

#mopology 
kernel_o = np.ones((5,5),np.uint8)
#kernel_c = np.ones((1,1),np.uint8)
morp_im = cv2.morphologyEx(otsu_im,cv2.MORPH_OPEN,kernel_o) 
#morp_im = cv2.morphologyEx(otsu_im,cv2.MORPH_CLOSE,kernel_c)

print morp_im.shape 

#extract tissue patch 

x_l,y_l = morp_im.nonzero() 

x_ws=(x_l * slide.level_downsamples[slide_level]).astype(int) 
y_ws=(y_l * slide.level_downsamples[slide_level]).astype(int) 

patch = slide.read_region((x_ws[0],y_ws[0]),0,(64,64)) 
patch = np.array(patch) 
rgb_patch = cv2.cvtColor(patch,cv2.COLOR_RGBA2RGB) 

plt.imshow(rgb_patch) 
plt.title("patch image") 
plt.axis('off') 


plt.show() 



"""
#image show !!
plt.subplot(121)
#plt.imshow(morp_im,cmap ="gray")
plt.imshow(wsi_mask,cmap="gray")
plt.title("WSI masked")
#plt.title("Morphology image")
plt.axis('off') 

plt.subplot(122)
plt.imshow(rgb_im)
plt.title("Rgb_image") 
plt.axis('off') 

plt.show()


cv2.imwrite("wsi.jpg",wsi_mask)
"""










