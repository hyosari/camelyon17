from os import listdir
from os.path import join, isfile, exists, splitext
import sys
import sys
import cv2
import numpy as np
from openslide import OpenSlide 
import matplotlib.pyplot as plt

from extract_xml import get_opencv_contours_from_xml,get_mask_from_opencv_contours,convert_contour_coordinate_resolution

x_pwd = "/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/lesion_annotations/"
tif_pwd = "/mnt/nfs/kyuhyoung/pathology/breast/camelyon17/training/centre_4"

filename ="patient_099_node_4.xml" #listdir(x_pwd)[0]
tif_filename =  splitext(filename)[0]+".tif"

print "file name is : "+filename 

#patient_number = (filename[8:11]).astype(int)
#print patient_number


xml_path = join(x_pwd,filename) 
tif_path = join(tif_pwd,tif_filename)

# confirm xml file exists 
if exists(xml_path):
    print "the path exists\n" 
else:
    print "the file does not exist\n"
     
#confirm tif file exists
if isfile(tif_path):
    slide = OpenSlide(tif_path) 
else:
    print "the tif path does not exist\n"
    

# contours from xml 

 






# slide show 
slide_level = 4
tile_0255 = slide.read_region((0,0),slide_level,slide.level_dimensions[slide_level])
im_rgba = np.array(tile_0255)
im_gray = cv2.cvtColor(im_rgba,cv2.COLOR_RGBA2GRAY)
im_rgb = cv2.cvtColor(im_rgba, cv2.COLOR_RGBA2RGB)
#get mask 
#mask_image = get_mask_from_opencv_contours(opencv_contours,slide,slide_level)
#print mask_image
#print np.nonzero(mask_image)

#temp = convert_contour_coordinate_resolution(opencv_contours,slide.level_downsamples[slide_level])
opencv_contours = get_opencv_contours_from_xml(xml_path,slide.level_downsamples[slide_level])
maske_image = get_mask_from_opencv_contours(opencv_contours,slide,slide_level)
#print len(temp)
cont_img = im_rgb
print 'cont_image dimension : ',cont_img.shape 
con =cv2.drawContours(cont_img,opencv_contours,-1,(0,255,0),-1)

_,anno,_ = cv2.split(cont_img) 
anno = anno == 255
anno = anno.astype(int) 
#anno = cv2.cvtColor(anno,cv2.COLOR_RGB2GRAY)

print anno.shape
print np.max(anno)
#plt.figure
plt.subplot(221)
plt.imshow(anno,cmap="gray")
plt.axis('off')
plt.subplot(222)
plt.imshow(cont_img)
plt.axis('off')
#cv2.imshow("GRAY", im_gray) 
#cv2.imshow("CONTOUR", cont_img)

#cv2.imshow("MASK",mask_image)
#cv2.imshow("RGBA",im_rgba)
#cv2.waitKey()
#cv2.destroyAllWindows()
plt.tight_layout()
plt.show()











