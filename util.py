""" 
    this file has the functions of preprocessing 
    
    otsu_thresholding 
    center_of_sliding_level
    connected_component_image

"""
import sys
import numpy as np 
from skimage.filters import threshold_otsu
import cv2,math
from skimage import measure


def otsu_thresholding(im_float):
    
    """
        apply otsu thresholding on the whole slide

        im_float = input image as float type 

        return otsued image with gray, otsu threshold value 
    """

    print("threshold_otsu\n")
    threshold_global_Otsu = threshold_otsu(im_float)

    #thresholding 
    im_bool= (im_float > threshold_global_Otsu)
    im_int = im_bool.astype(float)
    print im_int*255
    return im_int*255, threshold_global_Otsu

def center_of_slide_level(slide,level):
    
    """
        center x,y point of input level image 

        slide =  input slide with Openslide type 
        level = disired level 

        return  center point x, y 
    """

    c_x, c_y = slide.level_dimensions[level]
    c_x/=2
    c_y/=2
    return c_x, c_y 

def connected_component_image(otsu_image,component_num):

    """ 
        apply Connected Component Analysis to otsu_image 
        it is because of detect tissue 
        choose the label that has largest spces in the image

        otsu_image = input image that applied otsu thresholding
        component_num = threshold number of CCI(connected_component_image) 

        max_label = maximum label of components
        cnt_label = the number of pix which in lebel 
        argsort_label = sorted index of cnt_label list

        return tissue image 
    """

    image_labels = measure.label(otsu_image,background=0)

    
    max_label = np.max(image_labels)
    cnt_label = []

    print "before change componont num : {0}".format(component_num)
    #component number check ( shold compoenet number < maximum component of image ) 
    if component_num > max_label :
        component_num = max_label
    print "after change component num : {0}".format(component_num)

    for i in range(1,max_label+1):
        temp = (image_labels == i) 
        cnt_nonzero = np.count_nonzero(temp) 
        cnt_label.append(cnt_nonzero) 

    argsort_label = np.argsort(np.array(cnt_label))

    #tissue image initialize 
    tissue_image = np.zeros(image_labels.shape) 
    
    for i in range(component_num):
        temp= (image_labels == argsort_label[i])
        temp= temp.astype(int)
        tissue_image += temp 
    
    tissue_image = tissue_image.astype(float) 

    return tissue_image 


