from openslide import OpenSlide
from os import listdir
from os.path import join, isfile, exists, splitext
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from util import otsu_thresholding
from image_plot import subplot_show
from extract_xml import get_opencv_contours_from_xml
from skimage.transform.integral import integral_image, integrate
from integral import patch_sampling_using_integral


pwd ="/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/centre_4/"
filename = "patient_099_node_4.tif" 

xml_pwd ="/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/lesion_annotations/"
xml_name = "patient_099_node_4.xml"


slide_path = join(pwd,filename) 
xml_path = join(xml_pwd,xml_name) 

"""Open tif file !! """

slide = OpenSlide(slide_path) 

print "WSI Dimensions (width , hight) : {0}".format(slide.dimensions) 


"""Slide.read reagion""" 

slide_level=3
rgba_im = slide.read_region((0,0),slide_level,slide.level_dimensions[slide_level])
rgba_im= np.array(rgba_im)

rgb_im = cv2.cvtColor(rgba_im,cv2.COLOR_RGBA2RGB) 

"""Slide image convert to gray"""

gray_im = cv2.cvtColor(rgb_im,cv2.COLOR_RGB2GRAY) 

"""Slide image apply atsu thresholding"""

otsu_im,o_th = otsu_thresholding(gray_im)

"""Slide image Draw Contours"""
#cont_im = rgb_im 
#cv2.drawContours(cont_im,otsu_im,-1,(0,255,0),-1) 

"""mopology """

kernel_o = np.ones((5,5),np.uint8)
#kernel_c = np.ones((1,1),np.uint8)
morp_im = cv2.morphologyEx(otsu_im,cv2.MORPH_OPEN,kernel_o) 
#morp_im = cv2.morphologyEx(otsu_im,cv2.MORPH_CLOSE,kernel_c)

print morp_im.shape 

morp_im = morp_im == 0 
morp_im = (morp_im).astype(float)

"""patch size,leveled patch size """

patch_size = 256
leveled_patch_size = int(patch_size/slide.level_downsamples[slide_level])

""" patch random pick:tissu 

    extract tissue patch



x_l,y_l = morp_im.nonzero() 

x_ws=(np.round(x_l * slide.level_downsamples[slide_level])).astype(int) 
y_ws=(np.round(y_l * slide.level_downsamples[slide_level])).astype(int)


print "patch size:",patch_size

patch_num = 10
rand_point = np.random.randint(len(x_ws),size=patch_num) 

rgb_patch =[]

for i in range(patch_num):
    patch = slide.read_region((y_ws[rand_point[i]],x_ws[rand_point[i]]),0,(patch_size,patch_size)) 
    patch=np.array(patch) 
    rgb_patch.append(cv2.cvtColor(patch,cv2.COLOR_RGBA2RGB)) 
    

print "x: {} y: {}".format(x_ws[i],y_ws[i])


#subplot_show(rgb_patch,2,5,"tissue")
"""

"""" patch random pick : tummor

    # Get tummor contours from xml

"""

tummor_contours = get_opencv_contours_from_xml(xml_path,slide.level_downsamples[slide_level])
tum_im =rgb_im.copy() 
cv2.drawContours(tum_im,tummor_contours,-1,(0,255,0),-1) 

## Get Mask
_,tum,_ = cv2.split(tum_im) 
tum = tum == 255 
tum = tum.astype(int) 

tx_l,ty_l = tum.nonzero()

sampling_num =10

tum_patch_list , tum_patch_point = patch_sampling_using_integral(slide,slide_level,tum,patch_size,sampling_num)




subplot_show(tum_patch_list,2,5,"tummor") 

""" get Normal mask 

     tissu mask(morp_im) - tummor mask(tum) = normal mask 

"""

normal_im = morp_im - tum  ## np.min(normal_im) := -1.0 
normal_im = normal_im == 1.0 
normal_im = (normal_im).astype(int) 

nx_l,ny_l = normal_im.nonzero()

nor_patch_list , nor_patch_point = patch_sampling_using_integral(slide,slide_level,normal_im,patch_size,sampling_num)
subplot_show(nor_patch_list,2,5,"normal") 


"""
draw point that we picked on the level slide image 

    normal point(slide_level) = (x_l,y_l) ,tummor point(slide_level) = (tx_l,ty_l) 

"""

rand_tummor_point = np.array(tum_patch_point)
rand_normal_point = np.array(nor_patch_point)



#wh_pix = 20
for p_x,p_y in rand_normal_point:
    cv2.circle(rgb_im,(p_y,p_x),100,(0,255,0),-1)
    #cv2.rectangle(rgb_im,point,(point[0]+wh_pix,point[1]+wh_pix),(0,255,0),20) 
for p_x,p_y in rand_tummor_point:
    cv2.circle(rgb_im,(p_y,p_x),100,(255,0,0),-1)
    #cv2.rectangle(rgb_im,point,(point[0]+wh_pix,point[1]+wh_pix),(255,0,0),20)

#WSI show !  
plt.figure("point")
plt.imshow(rgb_im) 
plt.axis("off") 

# tummor contour show! 

plt.figure("tummor contour")
plt.imshow(tum_im)
plt.axis("off")






plt.show()
















