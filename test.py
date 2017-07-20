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

pwd = "/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/centre_4/"

filename ="patient_099_node_4.tif" #listdir(pwd) [4]
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
slide_level = 7
s_level_w, s_level_h = slide.level_dimensions[slide_level]
print "slide.level_count-1 dimensions (width,heigth):{0}\n".format(slide.level_dimensions[slide_level])
#center of slide
c_x, c_y = center_of_slide_level(slide,slide_level)
print "center of x : {0} , center of y : {0}".format(c_x,c_y)

#slide padding size 
padding = 10 
#read_region
tile_0255 = slide.read_region((0,0),slide_level,(s_level_w,s_level_h)) 
im_rgba = np.array(tile_0255)

#convert to gray from rgba 
im_gray=cv2.cvtColor(im_rgba,cv2.COLOR_RGBA2GRAY)
im_rgb = cv2.cvtColor(im_rgba,cv2.COLOR_RGBA2RGB)
#apply atsu thresholding
im_gray_ostu,x = otsu_thresholding(im_gray)
temp = im_gray
#cv2.drawContours(im_gray,im_gray_ostu,-1,255,-1)
#cv2.findContours(im_gray,)
print "ostu image  max value : {0}".format(np.max(im_gray_ostu))
print "otsu threaholding :{0}\n".format(x)
#image_labels = measure.label(im_gray_ostu)
print "image_labels \n"
print  "gray image shape:", im_gray.shape


#apply mopology 
kernel = np.ones((2,2),np.uint8)
kernel_1 = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel)
opening_1 = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel_1)
closing = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_CLOSE,kernel)

print opening_1

#image show !!!
plt.subplot(221)
#plt.imshow(im_gray_ostu,cmap='gray')
plt.imshow(opening,cmap='gray')
plt.title("opening ker_2")
#plt.imshow(im_rgb)
#plt.imshow(im_gray,cmap='gray')
plt.axis('off')

plt.subplot(222)
plt.imshow(opening_1,cmap="gray")
plt.title("opening ker_5")
plt.axis('off')

plt.subplot(223)
plt.imshow(im_gray_ostu,cmap='gray')
plt.title("ostu")
plt.axis('off')

plt.subplot(224)
plt.imshow(closing,cmap ='gray')
plt.title("closing")
plt.axis('off')
#plt.show()

#cv2.imshow("gray",im_gray)
#cv2.imshow("RGB",im_rgba)
#cv2.imshow("Otsu",im_gray_ostu)
#cv2.imshow("label",image_labels)
cv2.imshow("opening",opening_1)
cv2.waitKey()
#cv2.destoryAllWindows()
cv2.imwrite("tissue.jpg",opening_1)
#cv2.imwrite("WSI.jpg",im_rgb)
