import cv2 
import numpy as np 
from PIL import Image 
from skimage.transform.integral import integral_image, integrate 
from random import randint 



def patch_sampling_using_integral(slide,slide_level,mask,patch_size,patch_num):

    """
    patch sampling on whole slide image 

    slide = OpenSlide Object 
    slide_level = level of mask 
    mask = mask image ( 0-1 int type nd-array)
    patch_size = size of patch scala integer n 
    patch_num = the number of output patches 

    return list of patches(RGB Image), list of patch point(left top)  
    
    """

    x_l,y_l = mask.nonzero() 
    level_patch_size = int(patch_size/slide.level_downsamples[slide_level])

    x_ws = (np.round(x_l*slide.level_downsamples[slide_level])).astype(int)
    y_ws = (np.round(y_l*slide.level_downsamples[slide_level])).astype(int)

    patch_list = []
    patch_point =[]

    while(len(patch_list) < patch_num) :
        #random Pick point in mask 
        p_idx = randint(0,len(x_l)-1)
        #Get the point in mask 
        level_point_x,level_point_y = x_l[p_idx], y_l[p_idx]
        #Check boundary to make patch 
        check_bound = np.resize(np.array([level_point_x+level_patch_size,level_point_y+level_patch_size]),(2,))

        if check_bound[0] > mask.shape[0] or check_bound[1] > mask.shape[1]:
            continue
        #make patch from mask image
        level_patch_mask = mask[int(level_point_x):int(level_point_x+level_patch_size),int(level_point_y):int(level_point_y+level_patch_size)]
        
        #apply integral 
        ii_map = integral_image(level_patch_mask)
        ii_sum = integrate(ii_map,(0,0),(level_patch_size-1,level_patch_size-1))
        area_percent = float(ii_sum)/(level_patch_size**2)

        if area_percent<0.8:
            continue 
        
        #patch,point is appended the list 
        print "region percent: ",area_percent
        patch_point.append((x_l[p_idx],y_l[p_idx]))
        patch=slide.read_region((y_ws[p_idx],x_ws[p_idx]),0,(patch_size,patch_size))
        patch =np.array(patch) 

        patch_list.append(cv2.cvtColor(patch,cv2.COLOR_RGBA2RGB))

    return patch_list, patch_point 
