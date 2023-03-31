import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
from tqdm import tqdm
import cv2

coco_category_to_id_v1 = { 'aeroplane':1,
    'bicycle':2,
    'bird':3,
    'boat':4,
    'bottle':5,
    'bus':6,
    'car':7,
    'cat':8,
    'chair':9,
    'cow':10,
    'diningtable':11,
    'dog':12,
    'horse':13,
    'motorbike':14,
    'person':15,
    'pottedplant':16,
    'sheep':17,
    'sofa':18,
    'train':19,
    'tvmonitor':20}

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False,classes="person",categories=None,dataset=None):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    def compare(start,step,TP,P,T,input_type,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder,name.split("/")[-1]).replace("gtFine_labelTrainIds","leftImg8bit")
#                 print(os.path.exists(predict_file))
                if not os.path.exists(predict_file):
                    continue
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file) or predict.sum()>(512*512*0.9)
                
                if dataset == "VOC":
                    if classes=="person":
                        if predict.shape[0]==4 or predict.sum()>(512*512*coco_category_to_id_v1[classes]*0.5) or predict.sum()<(512*512*coco_category_to_id_v1[classes]*0.05):
                            continue
                        pass
                    elif classes=="diningtable":
                        if predict.shape[0]==4 or predict.sum()>(512*512*coco_category_to_id_v1[classes]*0.9) or predict.sum()<(512*512*coco_category_to_id_v1[classes]*0.05):
                            continue
                        pass
                    elif classes=="pottedplant":
                        if predict.shape[0]==4 or predict.sum()>(512*512*coco_category_to_id_v1[classes]*0.94) or predict.sum()<(512*512*coco_category_to_id_v1[classes]*0.05):
                            continue
                        pass
                    elif classes=="sofa":
                        if predict.shape[0]==4 or predict.sum()>(512*512*coco_category_to_id_v1[classes]*0.94) or predict.sum()<(512*512*coco_category_to_id_v1[classes]*0.05):
                            continue
                        pass
                    elif classes=="tvmonitor":
                        # tv tvmonitor
                        mask = 1*(predict!=0)
                        mask = np.array(mask, np.uint8)
                        if len(np.unique(mask))==1:
                            continue
                        contours = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  
                        cnts = contours[0]
                        ww,hh=0,0
                        for cnt in cnts:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w*h>ww*hh:
                                ww=w
                                hh=h
                        if ww/hh>2.5 or ww/hh<1:
                            continue
                        if mask.sum()/(ww*hh)<0.7:
                            continue
#                 elif classes=="boat":
#                     if predict.shape[0]==4 or predict.sum()>(512*512*4*0.5) or predict.sum()<(512*512*4*0.05):
#                         continue
#                     pass
#                 else:
#                     if predict.shape[0]==4:
#                         continue
                
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder,'%s.npy'%name.split("/")[-1].replace(".png",""))

#                 predict_dict = np.load(predict_file, allow_pickle=True).item()
#                 h, w = list(predict_dict.values())[0].shape
#                 tensor = np.zeros((21,h,w),np.float32)
#                 for key in predict_dict.keys():
#                     tensor[key+1] = predict_dict[key]

                if not os.path.exists(predict_file):
                    continue    
                tensor = np.load(predict_file, allow_pickle=True)
                
                if len(tensor.shape) == 2:
                    continue
                    
                tensor[0,:,:] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
            
            
            gt_file = os.path.join(gt_folder,name)
        
            gt = np.array(Image.open(gt_file))
            

            if dataset == "Cityscapes" and 0:
#             Vehicle     13: 'car', 14: 'truck', 15: 'bus',  16: 'train',
#             baseline: 40  -fp: 54.7  -fn: 47.6   67.7
                gt[gt==13]=13
                gt[gt==14]=13
                gt[gt==15]=13
                gt[gt==16]=13

                predict[predict==13]=13
                predict[predict==14]=13
                predict[predict==15]=13
                predict[predict==16]=13

                # Human     11: 'person', 12: 'rider',
                predict[predict==11]=12
                gt[gt==11]=12

                # motorbike
                gt[gt==19]=255
                gt[gt==18]=255
            
#             gt[gt!=11]=0
#             predict[predict!=11]=0
            
#             predict[predict==14]=0
#             predict[predict==116]=14
#             gt[gt==12]=11
#             gt[gt==19]=255
#             gt[gt==18]=255

              # VOC
#             predict[predict==12]=15
#             predict[predict==11]=15
#             predict[predict==20]=7
#             predict[predict==116]=14
#             gt[gt==17]=255

#             gt[(gt!=11)*(gt!=255)]=0
#             predict[(predict!=11)*(gt!=255)]=0
            
#             gt[gt==17]=12
#             gt[(gt==18)*(predict==11)]=11
#             gt[(gt==17)*(predict==11)]=11
            
            # Ride     17: 'motorcycle', 18: 'bicycle'
#             gt[gt==17]=18
#             predict[predict==18]=14
            
#             if 17 in np.unique(predict) and 17 not in np.unique(gt):
#                 print(name.split("/")[-1])
#                 continue
#                 print(name.split("/")[-1])
                
#             if 17 in np.unique(gt) and 17 not in np.unique(predict):
#                 print(name.split("/")[-1])
#                 continue
#                 print(name.split("/")[-1])    
#             gt[gt==18]=255
            cal = (gt<255) 
#             print(np.unique(predict),np.unique(gt))
#             predict[predict==15]=16
#             gt[gt==16]=15
#             gt[gt==17]=15
            mask = (predict==gt) * cal
# #             if TP[0].value<0 or T[0].value<0 or P[0].value<0:
# #                 print(gt_file)
# #                 assert False
#             print(TP[1].value,T[1].value,P[1].value)
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 

    for i in range(num_cls):
        
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    
    if dataset == "VOC" and 0:
        seen =0
        unseen = 0
        for i in range(1,16):
            seen+=IoU[i]
        for i in range(16,21):
            unseen+=IoU[i]
        seen_iou = seen/15
        unseen_iou = unseen/5
        print("seen:",seen_iou)
        print("unseen:",unseen_iou)
        print("harmonic mean:",2 * seen_iou * unseen_iou / (seen_iou + unseen_iou))
        
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
#     loglist['mIoU'] = miou * 100

#  VOC   car 7  person 15 
#  cityscapes  bus:15  person: 11   bottle
  #coco2017: person:1
    if IoU[coco_category_to_id_v1[classes]]*100!=0:
        print(name_list)
        print('{}:{}'.format(categories[coco_category_to_id_v1[classes]],IoU[coco_category_to_id_v1[classes]]*100))
        
    if printlog:
        for i in range(num_cls):
            print(i)
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()

def writedict_txt(filepath, dictionary):
    logfile = open(filepath,'a')
    for s in dictionary:
        s += '\n'
        logfile.write(s)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='./VOC2012/ImageSets/Segmentation/train.txt', type=str)
    parser.add_argument("--predict_dir", default='./out_rw', type=str)
    parser.add_argument("--output_dir", default='./out_rw', type=str)
    parser.add_argument("--dataset", default='VOC', type=str)
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='./evallog.txt',type=str)
    parser.add_argument('--comment', default="weijia", type=str)
    parser.add_argument('--classess', default="person", type=str)
    parser.add_argument('--type', default='npy', choices=['npy', 'png'], type=str)
    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--t', default=None, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--select', default=False, type=bool)
    args = parser.parse_args()

    if args.type == 'npy':
        assert args.t is not None or args.curve
    
    debug = False
    if args.dataset != "Cityscapes" or not debug:
        name_list = [i for i in os.listdir(args.gt_dir) if "png" in i]
        check_name_list = [i for i in os.listdir(args.predict_dir) if "png" in i]
    else:
        name_list = []
        check_name_list = []
        for el in os.listdir(args.gt_dir):
            cls_path = os.path.join(args.gt_dir,el)
            for image_one in os.listdir(cls_path):
#                 print(image_one)
                if "gtFine_labelTrainIds" in image_one:
                    name_list.append(el+"/"+image_one)
#                     print(image_one)
#                     check_name_list.append(el+"/"+image_one.replace("leftImg8bit.png","gtFine_labelTrainIds.png"))
                    
    print("dataset: {}, class: {}".format(args.dataset,args.classess))
    print(args.select)
    
    
#     check = []
#     with open("./DiffSeg_Data/VOC_Multi_Attention_car_sub_20000_GPT3/selected_90.txt", "r") as f:
#         data = f.readlines()
#     for d in data:
#         check.append(d.replace("\n",""))

    # VOC:21  ADE150
    if args.dataset == "VOC":
        class_number = 21
        categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
                  'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    elif args.dataset == "ADE":
        class_number = 150
        categories = [
        "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
        "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
        "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
        "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
        "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
        "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
        "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
        "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
        "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
        "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
        "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
        "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
        "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
        "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
        "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
        "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
        "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
        "glass", "clock", "flag"
        ]
    elif args.dataset == "Cityscapes":
        class_number = 19
        #'background',
        categories = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain',
                  'sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
    elif args.dataset == "COCO2017":
        class_number = 81
        categories = ['background','person','bicycle','car','motorcycle','airplane', 'bus','train', 'truck','boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog','horse','sheep', 'cow','elephant','bear','zebra',
 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee','skis', 'snowboard', 'sports ball', 'kite',
'baseball bat', 'baseball glove','skateboard', 'surfboard', 'tennis racket', 'bottle','wine glass', 'cup', 'fork', 'knife','spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
'carrot','hot dog', 'pizza','donut','cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse','remote', 'keyboard','cell phone','microwave', 'oven','toaster','sink', 'refrigerator',
'book', 'clock', 'vase','scissors', 'teddy bear','hair drier','toothbrush'
]
    
    
    if args.select:
        selected_files_dict = {}
        selected_files = []
        scores = []
        if debug:
            name_list_ = name_list
            check = []
            with open("./DiffMask_VOC/VOC_Multi_Attention_bird_sub_30000_NoClipRetrieval/select_92.0.txt", "r") as f:
                data = f.readlines()
            for d in data:
                check.append(d.replace("\n",""))
#             name_list = [i for i in name_list if i in check]
            name_list_ = [i for i in check_name_list if i in name_list]
#             name_list_ = [i for i in check_name_list if i in name_list][:3000]
        else:
            name_list_ = [i for i in check_name_list if i in name_list]
#         name_list_ = [i for i in name_list_ if i in check]
        
        for i in tqdm(name_list_[:20000]):
            
            name_list = [i]
            if not args.curve:
                loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, class_number, args.type, args.t, printlog=False, classes=args.classess,categories=categories,dataset=args.dataset)
            
                scores.append(str(loglist[args.classess]))
                
                if not debug:
                    selected_files_dict[i]=loglist[args.classess]
                    
#                     if loglist[args.classess]>args.threshold:
#                         selected_files.append(i)
        selected_files_dict = sorted(selected_files_dict.items(), key=lambda x:x[1], reverse=True)
        selected_files_dict = dict(selected_files_dict[:int(len(selected_files_dict)*(1-args.threshold))])
        for c in selected_files_dict:
#             print(selected_files_dict[c])
            selected_files.append(c)

        #             writelog(args.logfile, loglist, args.comment+"  "+ i)
#             else:
#                 l = []
#                 for i in range(40,60):
#                     t = i/100.0
#                     loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21, args.type, t)
#                     l.append(loglist['mIoU'])
#                     print('%d/60 background score: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
#                 writelog(args.logfile, {'mIoU':l}, args.comment)
#         writedict_txt('{}/select_score_after.txt'.format(args.output_dir,args.threshold),scores)


        if not debug:
#             writedict_txt('{}/select_score_{}.txt'.format(args.output_dir,args.threshold),scores)
            writedict_txt('{}/select_{}.txt'.format(args.output_dir,args.threshold),selected_files)
    else:
        check = []
        with open("./DiffMask_VOC/VOC_Multi_Attention_cat_sub_30000_NoClipRetrieval/select_92.0.txt", "r") as f:
            data = f.readlines()
        for d in data:
            check.append(d.replace("\n",""))
        
        print(len(check))

        if debug:
            name_list = name_list
            #  munster_000030_000019_gtFine_labelTrainIds.png
#             name_list = ["munster/munster_000040_000019_gtFine_labelTrainIds.png","munster/munster_000030_000019_gtFine_labelTrainIds.png"]
#munster_000128_000019_leftImg8bit.png
#             name_list_ = [i for i in check_name_list if i in name_list]
#             name_list = [i for i in name_list if i!="munster/munster_000128_000019_gtFine_labelTrainIds.png" and i!="frankfurt/frankfurt_000001_027325_gtFine_labelTrainIds.png"]
#         ""
#             name_list = [i for i in name_list if i!="2010_001553.png" and i!="2008_005915.png"]
        else:
            name_list = [i for i in check_name_list if i in name_list]
#         print(len(name_list))    
#         name_list = [i for i in name_list if i in check]
        
        print(len(name_list))
        if not args.curve:
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, class_number, args.type, args.t, printlog=True, classes=args.classess,categories=categories,dataset=args.dataset)
        else:
            l = []
            for i in range(40,60):
                t = i/100.0
                loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, class_number, args.type, t)
                l.append(loglist['mIoU'])
                print('%d/60 background score: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
            writelog(args.logfile, {'mIoU':l}, args.comment)
