from os.path import join, isfile, exists, splitext
from os import listdir
from openslide import OpenSlide
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
from image_plot import subplot_show 
from integral import patch_sampling_using_integral,tumor_patch_sampling_using_centerwin


def filelist_in_directory(fdir):
    """ retrun file name list with sorted """
    f_list= [ f for f in listdir(fdir)] 
    f_list.sort() 

    return f_list



def patch_generation(tif_dir,mask_dir,save_fdir,save_pdir,save_cdir,slide_level,patch_size,patch_num,tumor_win=False):
    """
     save patch image and extraction point with csv, jpg image on the directory

     tif_dir : the tif directory
     mask_dir : the mask directory 
     save_fdir : patch saved directory
     save_pdir : point jpg directory 
     save_cdir : Y,X csv directory 
     slide : slide_level that mask image was applied with 
     patch_size : patch size square 
     patch_num : the number of patches in a whole slide 
     tumor_win : tumor window patch sampling true
     
    
        tif file and mask file sholud be one to one mached and same ordered 
    """

    list_tif_name = filelist_in_directory(tif_dir)
    list_mask_name = filelist_in_directory(mask_dir) 

    for slide_idx in range(len(list_tif_name)): 
        pwd_tif = join(tif_dir,list_tif_name[slide_idx])
        pwd_msk = join(mask_dir,list_mask_name[slide_idx]) 
        filename = splitext(list_tif_name[slide_idx])[0]
        
        # open slide, csv, BGR_image, mask
        slide = OpenSlide(pwd_tif) 
        f = open(save_cdir+filename+".csv",'wt')
        c_writer = csv.writer(f)
        c_writer.writerow(('Y','X'))
        rgba_pil = slide.read_region((0,0),slide_level,slide.level_dimensions[slide_level])
        bgr_im = cv2.cvtColor(np.array(rgba_pil),cv2.COLOR_RGBA2BGR)
        mask = cv2.imread(pwd_msk,cv2.IMREAD_GRAYSCALE)

        #if mask tunes 255 change to 1 
        if np.max(mask) == 255:
            mask = mask == 255
            mask = mask.astype(int)

        # sampling patches
        if tumor_win:
            patch_list,patch_point = tumor_patch_sampling_using_centerwin(slide,slide_level,mask,patch_size,patch_num)
        else:
            patch_list,patch_point = patch_sampling_using_integral(slide,slide_level,mask,patch_size,patch_num)
        p_l_size = patch_size/ slide.level_downsamples[slide_level] 
        p_l_size = int(p_l_size)

        #image wirte and save patches 
        for f_th in range(len(patch_list)): 
            cv2.imwrite(save_fdir+filename+"_patch_"+str(f_th)+"_"+str(patch_point[f_th][1])+"_"+str(patch_point[f_th][0])+"_"+str(patch_size)+".jpg",patch_list[f_th]) 

            c_writer.writerow((patch_point[f_th][1],patch_point[f_th][0]))
            cv2.rectangle(bgr_im,(patch_point[f_th][1],patch_point[f_th][0]),(patch_point[f_th][1]+p_l_size,patch_point[f_th][0]+p_l_size),(255,0,0),1)

        cv2.imwrite(save_pdir+filename+"_selection_point.jpg",bgr_im)
        print "complete patch extraction about "+ list_tif_name[slide_idx]
        f.close()

    return 0 












def patch_generation_normal_from_tumor(tif_dir,mask_dir,save_fdir,save_pdir,save_cdir,slide_level,patch_size,patch_num,sel_dir,tumor_win=False):
    """
     save patch image and extraction point with csv, jpg image on the directory

     tif_dir : the tif directory
     mask_dir : the mask directory 
     save_fdir : patch saved directory
     save_pdir : point jpg directory 
     save_cdir : Y,X csv directory 
     slide : slide_level that mask image was applied with 
     patch_size : patch size square 
     patch_num : the number of patches in a whole slide 
     tumor_win : tumor window patch sampling true
     
    
        tif file and mask file sholud be one to one mached and same ordered 
    """

    list_tif_name = filelist_in_directory(tif_dir)
    list_mask_name = filelist_in_directory(mask_dir) 
    list_sel_name = filelist_in_directory(sel_dir) 
    list_sel_name = list_sel_name[1::2]

    for slide_idx in range(len(list_tif_name)): 
        if slide_idx not in np.array([15,18,20,29,33,44,46,51,54,55,79,92,95])-1:
            pwd_tif = join(tif_dir,list_tif_name[slide_idx])
            pwd_msk = join(mask_dir,list_mask_name[slide_idx]) 
            filename = splitext(list_tif_name[slide_idx])[0]
            # open slide, csv, BGR_image, mask
            slide = OpenSlide(pwd_tif) 
            f = open(save_cdir+filename+".csv",'wt')
            c_writer = csv.writer(f)
            c_writer.writerow(('Y','X'))
            #rgba_pil = slide.read_region((0,0),slide_level,slide.level_dimensions[slide_level])
            #bgr_im = cv2.cvtColor(np.array(rgba_pil),cv2.COLOR_RGBA2BGR)
            ### change bgr_im 
            bgr_im = cv2.imread(sel_dir+list_sel_name[slide_idx])
            print sel_dir+list_sel_name[slide_idx]
            mask = cv2.imread(pwd_msk,cv2.IMREAD_GRAYSCALE)
            
            #if mask tunes 255 change to 1 
            if np.max(mask) == 255:
                mask = mask == 255
                mask = mask.astype(int)
            # sampling patches
            if tumor_win:
                patch_list,patch_point = tumor_patch_sampling_using_centerwin(slide,slide_level,mask,patch_size,patch_num)
            else:
                patch_list,patch_point = patch_sampling_using_integral(slide,slide_level,mask,patch_size,patch_num)
            #leveled patchsize 
            p_l_size = patch_size/ slide.level_downsamples[slide_level] 
            p_l_size = int(p_l_size)
            #image wirte and save patches 
            
            for f_th in range(len(patch_list)): 
                cv2.imwrite(save_fdir+filename+"_patch_"+str(f_th)+"_"+str(patch_point[f_th][1])+"_"+str(patch_point[f_th][0])+"_"+str(patch_size)+".jpg",patch_list[f_th]) 
                
                c_writer.writerow((patch_point[f_th][1],patch_point[f_th][0]))
                cv2.rectangle(bgr_im,(patch_point[f_th][1],patch_point[f_th][0]),(patch_point[f_th][1]+p_l_size,patch_point[f_th][0]+p_l_size),(0,0,255),1)
            cv2.imwrite(save_pdir+filename+"_selection_point.jpg",bgr_im)
            print "complete patch extraction about "+ list_tif_name[slide_idx]
            f.close()
                
    return 0 
