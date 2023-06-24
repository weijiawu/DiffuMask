import numpy as np
import os
import argparse
from PIL import Image
import pandas as pd
import multiprocessing
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

voc12_category_to_id_v1 = { 'aeroplane':0,
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

Cityscapes_category_to_id_v1 = { 'road':0,
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_list", default="./voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--dataset", default='/VOC2012', type=str)
    parser.add_argument("--cam_dir", default=None, type=str)
    parser.add_argument("--class_number", default=21, type=int)
    parser.add_argument("--classes", default="person", type=str)
    parser.add_argument("--out_crf", default=None, type=str)
    parser.add_argument("--crf_iters", default=10, type=float)
    parser.add_argument("--alpha", default=4, type=float)

    args = parser.parse_args()

    assert args.cam_dir is not None

    if args.out_crf:
        if not os.path.exists(args.out_crf):
            os.makedirs(args.out_crf)
    
#     args.infer_list = "./Diffusion/train_image"
    name_list = [i for i in os.listdir(args.infer_list) if "jpg" in i]
    print(len(name_list))
    
#     df = pd.read_csv(args.infer_list, names=['filename'])
#     name_list = df['filename'].values


    # https://github.com/pigcv/AdvCAM/blob/fa08f0ad4c1f764f3ccaf36883c0ae43342d34c5/misc/imutils.py#L156
    def _crf_inference(img, labels, t=10, n_labels=21, gt_prob=0.7):
        h, w = img.shape[:2]
        d = dcrf.DenseCRF2D(w, h, n_labels)
        U = unary_from_labels(labels, 21, gt_prob=gt_prob, zero_unsure=False)
        d.setUnaryEnergy(U)
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        
        feats = create_pairwise_bilateral(sdims=(60, 60), schan=(10, 10, 10),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=9,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        
#         feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
#                                           img=img, chdim=2)
#         d.addPairwiseEnergy(feats, compat=10,
#                             kernel=dcrf.DIAG_KERNEL,
#                             normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(t)

        return np.array(Q).reshape((n_labels, h, w))


    def _infer_crf_with_alpha(start, step, alpha):
        for idx in range(start, len(name_list), step):
            print(idx)
            name = name_list[idx]
            name = name.split("/")[-1].replace(".jpg","")
            cam_file = os.path.join(args.cam_dir, '%s.npy' % name)
            
#             cam_file = os.path.join(args.cam_dir, '%s.npy' % name)
            cam_dict = np.load(cam_file, allow_pickle=True).item()
            h, w = list(cam_dict.values())[0].shape
        
            tensor = np.zeros((args.class_number, h, w), np.float32)
            for key in cam_dict.keys():
                tensor[key + 1] = cam_dict[key]
            
#             tensor[0, :, :] = np.power(1 - np.max(tensor, axis=0, keepdims=True), alpha)
#             average = 
#             tensor[0, :, :] = np.power(1 - np.max(tensor, axis=0, keepdims=True), alpha)

            cam_dict_test = np.array(list(cam_dict.values()))
            
            if args.dataset == "VOC":
                target_map = cam_dict[int(voc12_category_to_id_v1[name.split("_")[1]])]
            elif args.dataset == "Cityscapes":
                target_map = cam_dict[int(Cityscapes_category_to_id_v1[name.split("_")[1]])]
            
            if args.classes == "person":
                high_threold = 55
                low_threold = 48
            elif args.classes == "sofa":
                high_threold = 58
                low_threold = 48
            elif args.classes == "aeroplane":
                high_threold = 53
                low_threold = 48
            elif args.classes == "boat":
                high_threold = 53
                low_threold = 48
            elif args.classes == "diningtable":
                high_threold = 64
                low_threold = 55
            else:
                high_threold = 53
                low_threold = 45
            

            min_number = 0
            threshold = low_threold*0.01
            for i in range(low_threold,high_threold,1):
                descresing_points = ((target_map>(i*0.01))*(target_map<((i+1)*0.01))).sum()
                if min_number < descresing_points:
                    min_number = descresing_points
                    threshold = i*0.01
            
            la_crf = ((target_map>threshold) * target_map).sum()/((target_map>threshold)*1).sum()
            ha_crf = ((target_map<threshold) * target_map).sum()/((target_map<threshold)*1).sum()
            
            if args.classes == "person":
                min_mmax = 0.01
            else:
                min_mmax = 0.06
                
            while ((target_map>la_crf)*1).sum() < (512*512)*min_mmax:
                if threshold<0.3:
                    break
                threshold = threshold - 0.03
                la_crf = ((target_map>threshold) * target_map).sum()/((target_map>threshold)*1).sum()


            while ((target_map<ha_crf)*1).sum() < (512*512)*min_mmax:
                if threshold>0.75:
                    break
                ha_crf = ((target_map<threshold) * target_map).sum()/((target_map<threshold)*1).sum()
                threshold = threshold + 0.03

            if alpha == "la":
                la_crf = ((target_map>threshold) * target_map).sum()/((target_map>threshold)*1).sum()
                if args.classes == "person":
                    la_crf = la_crf + 0.06
                elif args.classes == "aeroplane":
                    la_crf = la_crf - 0.02
                elif args.classes == "boat":
                    la_crf = la_crf - 0.02
                elif args.classes == "sofa":
                    la_crf = la_crf - 0.01
                elif args.classes == "diningtable":
                    la_crf = la_crf - 0.04
                else:
                    la_crf = la_crf - 0.04
                
                tensor[0, :, :] = la_crf
            else:
                ha_crf = ((target_map<threshold) * target_map).sum()/((target_map<threshold)*1).sum()
                if args.classes == "person":
                    ha_crf = ha_crf + 0.06
                elif args.classes == "aeroplane":
                    ha_crf = ha_crf + 0.05
                elif args.classes == "boat":
                    ha_crf = ha_crf + 0.05
                elif args.classes == "sofa":
                    ha_crf = ha_crf + 0.07
                elif args.classes == "diningtable":
                    ha_crf = ha_crf + 0.04
                else:
                    ha_crf = ha_crf + 0.02
                
                tensor[0, :, :] = ha_crf
            
            
            
            
            predict = np.argmax(tensor, axis=0).astype(np.uint8)
            img = Image.open(os.path.join(args.infer_list, name + '.jpg')).convert("RGB")
            img = np.array(img)
            crf_array = _crf_inference(img, predict)

#             crf_folder = args.out_crf + ('/%.2f' % alpha)
            crf_folder = args.out_crf + ('/{}'.format(alpha))
            if not os.path.exists(crf_folder):
                os.makedirs(crf_folder)

            np.save(os.path.join(crf_folder, name + '.npy'), crf_array)


    
#     alpha_list = [0.5, 1, 4, 8, 16, 24, 32]
#     alpha_list = [0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8, 0.9]  
    alpha_list = ["la","ha"]
#     alpha_list = [1, 2, 4, 8, 16, 24, 32]
    
    for alpha in alpha_list:
        p_list = []
        for i in range(8):
            p = multiprocessing.Process(target=_infer_crf_with_alpha, args=(i, 8, alpha))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        print(f'Info: Alpha {alpha} done!')
