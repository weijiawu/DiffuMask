import numpy as np
import os
import argparse
from PIL import Image
import pandas as pd
import multiprocessing
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import matplotlib.pyplot as plt
import cv2
import torch
# import clip
from PIL import Image
from tqdm import tqdm
import json
from random import choice
import numpy as np
import random
import shutil

coco_category_to_id_v1 = { 'aeroplane':0,
    'bicycle':1,
    'bird':2,
    'boat':3,
    'bottle':4,
    'bus':5,
    'car':6,
    'cat':7,
    'chair':8,
    'cow':9,
    'diningtable':10,
    'dog':11,
    'horse':12,
    'motorbike':13,
    'person':14,
    'pottedplant':15,
    'sheep':16,
    'sofa':17,
    'train':18,
    'tvmonitor':19}


def get_findContours(mask):
    
    idxx = np.unique(mask)
    if len(idxx)==1:
        return mask
    else:
        idxx = idxx[1]
    mask_instance = (mask>0.5 * 1).astype(np.uint8) 
    ontours, hierarchy = cv2.findContours(mask_instance.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓
    
    min_area = 0
    polygon_ins = []
    x,y,w,h = 0,0,0,0
    
    image_h, image_w = mask.shape[0:2]
    gt_kernel = np.zeros((image_h,image_w), dtype='uint8')
    for cnt in ontours:
        # 外接矩形框，没有方向角
        x_ins_t, y_ins_t, w_ins_t, h_ins_t = cv2.boundingRect(cnt)

        if w_ins_t*h_ins_t<1500:
            continue
        cv2.fillPoly(gt_kernel, [cnt], int(idxx))

    return gt_kernel


def aug(version="Augmentation_One",size="2",image_id=0):


#     version = "Augmentation_One"
    image_path = "./DiffSeg_Data/{}/train_image".format(version)
    mask_path = "./DiffSeg_Data/{}/ground_truth".format(version)

    image_list = [i for i in os.listdir(image_path) if "jpg" in i]
    image_list = [i for i in image_list if os.path.exists("./DiffSeg_Data/{}/ground_truth/{}".format(version,i.replace("jpg","png")))]

#     image_id = 6000
#     size=2
    for idx in tqdm(range(4000)):
        list_image = []
        list_mask = []
        for x in range(size):
            image_1 = choice(image_list)
            mask_1 = image_1.replace("jpg","png")


            img1 = cv2.imread("./DiffSeg_Data/{}/train_image/{}".format(version,image_1))
            mas1 = cv2.imread("./DiffSeg_Data/{}/ground_truth/{}".format(version,mask_1))


            for y in range(size-1):
                image_2 = choice(image_list)
                mask_2 = image_2.replace("jpg","png")


                img2 = cv2.imread("./DiffSeg_Data/{}/train_image/{}".format(version,image_2))
                mas2 = cv2.imread("./DiffSeg_Data/{}/ground_truth/{}".format(version,mask_2))

                img1 = np.concatenate([img1, img2], axis=1)
                mas1 = np.concatenate([mas1, mas2], axis=1)
            list_image.append(img1)
            list_mask.append(mas1)

        list_image_ha = list_image[0]
        list_mask_ha = list_mask[0]
        for i in range(1,size):

            list_image_ha = np.concatenate((list_image_ha, list_image[i])) 
            list_mask_ha = np.concatenate((list_mask_ha, list_mask[i])) 

    #     list_image = cv2.resize(list_image, (512, 512), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./DiffSeg_Data/{}/train_image_2/new_{}.jpg".format(version,image_id),list_image_ha)
        cv2.imwrite("./DiffSeg_Data/{}/ground_truth_2/new_{}.png".format(version,image_id),list_mask_ha)
        image_id+=1
    
    return image_id

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="Augmentation_One", type=str)
    parser.add_argument("--size", default=[1,2,3], type=int)
    parser.add_argument("--number", default=4000, type=int)
    parser.add_argument("--select_file", default="Augmentation_One", type=str)
    parser.add_argument("--TargetClassPath_Image", default="Augmentation_One", type=str)
    parser.add_argument("--TargetClassPath_Mask", default="Augmentation_One", type=str)
    parser.add_argument("--OtherClassPath", default="Augmentation_One", type=str)
    parser.add_argument("--OutputPath", default="Augmentation_One", type=str)
    args = parser.parse_args()

#     assert args.cam_dir is not None

#     if os.path.exists(args.OutputPath):
#         shutil.rmtree(args.OutputPath)
        
#     if not os.path.exists(args.OutputPath):
#         os.makedirs(args.OutputPath)
    
#     targe_class_imge_path = os.path.join(args.TargetClassPath,"train_image")
#     targe_class_imge_path = os.path.join(args.TargetClassPath,"train_image")
    
#     name_list = [i for i in os.listdir(args.infer_list) if "jpg" in i]
    
    image_id = 6000
    for size in [2,3]:
        image_id = aug(version="Augmentation_One",size=size,image_id=image_id)
    
    
   