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

def connected_component_image(otsu_image):

    """ 
        apply Connected Component Analysis to otsu_image 
        it is because of detect tissue 
        choose the label that has largest spces in the image

        otsu_image = input image that applied otsu thresholding

        max_label = maximum label of components
        cnt_label = the number of pix which in certin lebel 
        result_label = the label which indicate tissue 

        return tissue image 
    """

    image_labels = measure.label(otsu_image) 
    max_label = np.max(image_labels) 
    cnt_label = 0 
    result_label = 1

    for i in range(1,max_label):
        temp = (image_labels == i) 
        temp = temp.astype(float)
        cnt_nonzero = np.count_nonzero(temp) 
        if cnt_nonzero > cnt_label:
            cnt_label = cnt_nonzero
            result_label = i
    
    tissue_image = (image_labels == result_label)
    tissue_image = tissue_image.astype(float) 

    return tissue_image 


