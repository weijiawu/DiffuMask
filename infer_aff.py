import torch
import torchvision
from tool import imutils

import argparse
import importlib
import numpy as np

import voc12.data
from torch.utils.data import DataLoader
import scipy.misc
import torch.nn.functional as F
import os.path
import cv2

def get_indices_in_radius(height, width, radius):

    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius+1, radius):
            if x*x + y*y < radius*radius:
                search_dist.append((y, x))

    full_indices = np.reshape(np.arange(0, height * width, dtype=np.int64),
                              (height, width))
    radius_floor = radius-1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])

    indices_from_to_list = []

    for dy, dx in search_dist:

        indices_to = full_indices[dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_from_to = np.stack((indices_from, indices_to), axis=1)

        indices_from_to_list.append(indices_from_to)

    concat_indices_from_to = np.concatenate(indices_from_to_list, axis=0)

    return concat_indices_from_to


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

        if w_ins_t*h_ins_t<250:
            continue
        cv2.fillPoly(gt_kernel, [cnt], int(idxx))

    return gt_kernel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.vgg16_aff", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--cam_dir", required=True, type=str)
    parser.add_argument("--voc12_root", required=True, type=str)
    parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument("--out_rw", required=True, type=str)
    parser.add_argument("--beta", default=8, type=int)
    parser.add_argument("--logt", default=8, type=int)

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()

    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ImageDataset(args.infer_list, voc12_root=args.voc12_root,
                                               transform=torchvision.transforms.Compose(
        [np.asarray,
         model.normalize,
         imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    if not os.path.exists(args.out_rw):
        os.makedirs(args.out_rw)
    
    for iter, (name, img) in enumerate(infer_data_loader):

        name = name[0]
        print(iter)

        orig_shape = img.shape
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))
        
        cam = np.load(os.path.join(args.cam_dir, name + '.npy'),allow_pickle=True).item()
        cam_copy = np.array(list(cam.values()))
        cam_copy_idx = cam_copy.sum(axis=1).sum(axis=1)
        idx_ = np.nonzero(cam_copy_idx)[0][0]
        cam_copy = cam_copy[idx_]
        
        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k+1] = v
#         cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False))**args.alpha
        cam_full_arr[0] = 0.45
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        with torch.no_grad():
            args.beta = 16
            aff_mat = torch.pow(model.forward(img.cuda(), True), args.beta)
#             aff_mat = model.forward(img.cuda(), True)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(args.logt):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

            cam_vec = cam_full_arr.view(21, -1)
            
            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)
            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)

            
            _, cam_rw_pred = torch.max(cam_rw, 1)

            res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]

            
            res = get_findContours(res)
            cv2.imwrite(os.path.join(args.out_rw, name + '.png'), res)
            
