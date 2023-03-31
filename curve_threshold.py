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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_list", default="./voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='/VOC2012', type=str)
    parser.add_argument("--cam_dir", default=None, type=str)
    parser.add_argument("--out_crf", default=None, type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--crf_iters", default=10, type=float)
    parser.add_argument("--alpha", default=4, type=float)

    args = parser.parse_args()

    assert args.cam_dir is not None

    if args.out_crf:
        if not os.path.exists(args.out_crf):
            os.makedirs(args.out_crf)
    
    name_list = [i for i in os.listdir(args.infer_list) if "jpg" in i][:]
    
    if args.dataset == "voc":
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
    elif args.dataset == "cityscapes":
        coco_category_to_id_v1 = { 
        'road':0,
        'sidewalk':1,
        'building':2,
        'wall':3,
        'fence':4,
        'pole':5,
        'traffic light':6,
        'traffic sign':7,
        'vegetation':8,
        'terrain':9,
        'sky':10,
        'person':11,
        'rider':12,
        'car':13,
        'truck':14,
        'bus':15,
        'train':16,
        'motorcycle':17,
        'bicycle':18}
    
    def _crf_inference(img, labels, t=10, n_labels=21, gt_prob=0.5):
        h, w = img.shape[:2]
        d = dcrf.DenseCRF2D(w, h, n_labels)
        U = unary_from_labels(labels, 21, gt_prob=gt_prob, zero_unsure=False)
        d.setUnaryEnergy(U)
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(t)

        return np.array(Q).reshape((n_labels, h, w))


    def _infer_crf_with_alpha(start, step, alpha):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            name = name.split("/")[-1].replace(".jpg","")
            cam_file = os.path.join(args.cam_dir, '%s.npy' % name)
            
            ground_truth = os.path.join(args.cam_dir.replace("npy","refine_gt"), '%s.png' % name)

            cam_dict = np.load(cam_file, allow_pickle=True).item()
            h, w = list(cam_dict.values())[0].shape
        
            tensor = np.zeros((21, h, w), np.float32)
            for key in cam_dict.keys():
                tensor[key + 1] = cam_dict[key]

            cam_dict_test = np.array(list(cam_dict.values()))
            target_map = cam_dict[int(coco_category_to_id_v1[name.split("_")[1]])]

            roate = [i*0.01 for i in range(35,55)]
#             roate = [0.6]
            hhaa = []
            best_threhold = []
            
            if not os.path.isfile(ground_truth):
                continue
                
            gt = cv2.imread(ground_truth)[:,:,0]
            
            if os.path.isfile(os.path.join(args.out_crf, name + '.png')):
                continue
                
            max_iou = 0
            max_crf_array = 0
            best_threshold = 0
            for i in roate:
                tensor[0, :, :] = i
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
                img = Image.open(os.path.join(args.infer_list, name + '.jpg')).convert("RGB")
                img = np.array(img)
                crf_array = _crf_inference(img, predict)
                orig_image = img.copy()
                
                
                iddddd = int(coco_category_to_id_v1[name.split("_")[1]])+1
                crf_array = crf_array[iddddd]
                
#                 max_crf_array = 1*(crf_array.copy() >0.5)
#                 break
                gt_1 = 1*(gt.copy() == iddddd)
                crf_array_a = 1*(crf_array.copy() >0.5)
#                 res = get_findContours(crf_array_a)
                iou = (crf_array_a*gt_1).sum() /(crf_array_a.sum()+gt_1.sum()+1e-10) 

                if max_iou<iou:
                    max_iou = iou
                    best_threshold = i
                    max_crf_array = crf_array_a
                    
            print(best_threshold)
            if best_threshold == 0:
                max_crf_array = np.zeros((h, w), np.float32)
            best_threhold.append(best_threshold)
            res = get_findContours(max_crf_array*iddddd)
            
            
            cv2.imwrite(os.path.join(args.out_crf, name + '.png'), res)
        
        
#             plt.figure(figsize=(14,7)) #设置窗口大小
#             plt.subplot(1,3,1)
#             plt.imshow((tensor[int(coco_category_to_id_v1[name.split("_")[1]])+1])*255)
            
#             plt.subplot(1,3,2)
#             plt.imshow((res==iddddd)*255)

#             plt.subplot(1,3,3)
#             plt.imshow(img)

#             plt.savefig("./crf_debug/{}".format(name + '.jpg'))
#             plt.show()
#         print(np.array(best_threhold).mean())
    alpha_list = ["la"]

        
    for alpha in alpha_list:
        p_list = []
        for i in range(8):
            p = multiprocessing.Process(target=_infer_crf_with_alpha, args=(i, 8, alpha))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        print(f'Info: Alpha {alpha} done!')
