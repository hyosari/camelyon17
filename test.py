from openslide import OpenSlide
from os import listdir
from os.path import join, isfile, exists, splitext
import sys
import cv2
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
from util import otsu_thresholding, center_of_slide_level,connected_component_image
from skimage import measure 
from scipy import ndimage 

pwd = "/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/centre_0/"

filename = listdir(pwd) [3]
print "file name : "+filename+"\n"

slide_path = join(pwd,filename)

#Open tif File 
if isfile(slide_path):
    """is it file? """
    slide=OpenSlide(slide_path)
elif exists(slide_path):
    """ dose it exist? """
    print "slide_path :" + slide_path + " is not a readable file \n"
else:
    """ it is not a file and doesn't exist"""
    print "file dosen't exist in this path :"  + slide_path+"\n"

slide_w, slide_h = slide.dimensions
print "Whole Slide dimensions (with, heigth):{0}\n".format(slide.dimensions)


#Slid.level_dimensions
slide_level = slide.level_count -1 
s_level_w, s_level_h = slide.level_dimensions[slide_level]
print "slide.level_count-1 dimensions (width,heigth):{0}\n".format(slide.level_dimensions[slide_level])
#center of slide
c_x, c_y = center_of_slide_level(slide,slide_level)
print "center of x : {0} , center of y : {0}".format(c_x,c_y)

#slide padding size 
padding = 10 
#read_region
tile_0255 = slide.read_region((c_x,c_y),slide_level,(s_level_w,s_level_w)) 
im_rgba = np.array(tile_0255)

#convert to gray from rgba 
im_gray=cv2.cvtColor(im_rgba,cv2.COLOR_RGBA2GRAY)
#apply atsu thresholding
im_gray_ostu,x = otsu_thresholding(im_gray)

print "ostu image  max value : {0}".format(np.max(im_gray_ostu))
print "otsu threaholding :{0}\n".format(x)
#image_labels = measure.label(im_gray_ostu)
print "image_labels \n"

component_num = 1
image_labels = connected_component_image(im_gray_ostu,component_num)


#image show !!! 
plt.figure(figsize=(9,3.5))
plt.subplot(131)
plt.imshow(im_gray_ostu,cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(image_labels,cmap='spectral')
plt.axis('off')

plt.tight_layout()
plt.show()

#cv2.imshow("gray",im_gray)
cv2.imshow("RGB",im_rgba)
#cv2.imshow("Otsu",im_gray_ostu)
#cv2.imshow("label",image_labels)
cv2.waitKey()
cv2.destoryAllWindows()


