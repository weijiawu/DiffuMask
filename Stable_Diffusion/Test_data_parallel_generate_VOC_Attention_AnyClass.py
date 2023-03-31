from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import cv2
import json
import argparse
import multiprocessing as mp
import threading
from random import choice
import os
import argparse
from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality
from tqdm import tqdm
# import openai
#export OPENAI_API_KEY="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"

# openai.api_key = "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"
# openai.api_key = os.getenv("OPENAI_API_KEY")

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian



MY_TOKEN = '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'
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

coco_category_list_check_person = [    
    "arm",
    'person',
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "teenager"
]


VOC_category_list_check = {
    'aeroplane':['aerop','lane'],
    'bicycle':['bicycle'],
    'bird':['bird'],
    'boat':['boat'],
    'bottle':['bottle'],
    'bus':['bus'],
    'car':['car'],
    'cat':['cat'],
    'chair':['chair'],
    'cow':['cow'],
    'diningtable':['table'],
    'dog':['dog'],
    'horse':['horse'],
    'motorbike':['motorbike'],
    'person':coco_category_list_check_person,
    'pottedplant':['pot','plant','ted'],
    'sheep':['sheep'],
    'sofa':['sofa'],
    'train':['train'],
    'tvmonitor':['monitor','tv','monitor']
    }


coco_category_list_check = [    "arm",'aerop','lane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pot',
    'ted',
    'plant',
    'sheep',
    'sofa',
    'train',
    'tv',
    'monitor']

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


coco_category_list = [ 
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor']

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3,tokenizer=None,device=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
#         if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],tokenizer=None,device=None):
        super(AttentionControlEdit, self).__init__()
#         print(device)
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,tokenizer=None,device=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,tokenizer=tokenizer,device=device)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None,device=None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,device=device)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]],tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
#     print(values)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


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
from PIL import Image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    return out.cpu()


def mask_image(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0
    
        
    image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.2 + mask_3d_color[mask] * 0.8
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)  
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb

def get_findContours(mask):
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

#         if min_area<w_ins_t*h_ins_t:
#             min_area = w_ins_t*h_ins_t
#             x,y,w,h = x_ins_t, y_ins_t, w_ins_t, h_ins_t
# #             x_ins, y_ins, w_ins, h_ins = x_ins_t+min_x, y_ins_t+min_y, w_ins_t, h_ins_t

#             polygon_ins = cnt
#         print(w_ins_t*h_ins_t)
        if w_ins_t*h_ins_t<250:
            continue
        cv2.fillPoly(gt_kernel, [cnt], 1)

    return gt_kernel

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
    
    
def save_cross_attention(orignial_image,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put="./test_1.jpg",atten_put="",image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None):
    
#     ("up", "down")
#     ("mid", "up", "down")

    orignial_image = orignial_image.copy()
    show = True
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    
    # "up", "down"
    attention_maps_8s = aggregate_attention(attention_store, 8, ("up", "mid", "down"), True, select,prompts=prompts)
    attention_maps_8s = attention_maps_8s.sum(0) / attention_maps_8s.shape[0]
    
    
    attention_maps = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts)
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
    
    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts)
    attention_maps_32 = attention_maps_32.sum(0) / attention_maps_32.shape[0]
    
    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select,prompts=prompts)
    attention_maps_64 = attention_maps_64.sum(0) / attention_maps_64.shape[0]
    

    cam_dict = {}
#     for idx, class_one in enumerate(coco_category_list):

    gt_kernel_final = np.zeros((512,512), dtype='float32')
    number_gt = 0
    for i in range(len(tokens)):
        class_current = decoder(int(tokens[i])) 

        category_list_check = VOC_category_list_check[class_one]
        if class_current not in category_list_check:
            continue
#             if class_one != "person":
#                 if class_current not in coco_category_list_check:
#                     continue
#             else:
#                 if class_current not in coco_category_list_check_person:
#                     continue

        image_8 = attention_maps_8s[:, :, i]
        image_8 = cv2.resize(image_8.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_8 = image_8 / image_8.max()

        image_16 = attention_maps[:, :, i]
        image_16 = cv2.resize(image_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_16 = image_16 / image_16.max()

        image_32 = attention_maps_32[:, :, i]
        image_32 = cv2.resize(image_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_32 = image_32 / image_32.max()

        image_64 = attention_maps_64[:, :, i]
        image_64 = cv2.resize(image_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
        image_64 = image_64 / image_64.max()

        if class_one == "sofa" or class_one == "train" or class_one == "tvmonitor":
            image = image_8
        elif class_one == "diningtable":
            image = image_16
        else:
            image = (image_16 + image_32 + image_64) / 3
#             image = image_8

        gt_kernel_final += image.copy()
        number_gt += 1

    if number_gt!=0:
        gt_kernel_final = gt_kernel_final/number_gt

    id_ = coco_category_to_id_v1[class_one]
#     cam_dict[id_] = gt_kernel_final
    
    tensor = np.zeros((21, 512, 512), np.float32)
    tensor[id_ + 1] = gt_kernel_final
    
    
    tensor[0, :, :] = 0.45
    predict = np.argmax(tensor, axis=0).astype(np.uint8)
    crf_array = _crf_inference(orignial_image, predict) 
    crf_array = crf_array[id_ + 1]
    crf_array_a = (id_ + 1)*(crf_array.copy() >0.5)
    
    res = get_findContours(crf_array_a)
    cv2.imwrite(out_put, res)
    cv2.imwrite(atten_put,gt_kernel_final*255)
    


    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))
    
    
def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None,out_put = "",ldm_stable=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator, out_put = out_put)
        print("with prompt-to-prompt")
        
    

    images_here, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=generator, low_resource=LOW_RESOURCE)
    

    
    ptp_utils.view_images(images_here,out_put = out_put)
    return images_here, x_t



def sub_processor(pid , args):
    torch.cuda.set_device(pid)
    text = 'processor %d' % pid
    print(text)
    
    MY_TOKEN = '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'
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device)
    tokenizer = ldm_stable.tokenizer

    number_per_thread_num = int(int(args.image_number)/int(args.thread_num))
    
    image_cnt = pid * (number_per_thread_num*2) + 200000
    
    image_path = os.path.join(args.output,"train_image")
    mask_path = os.path.join(args.output,"mask")
    atten_path = os.path.join(args.output,"attention")
    
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(atten_path):
        os.makedirs(atten_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

  
    for rand in range(number_per_thread_num):
        g_cpu = torch.Generator().manual_seed(image_cnt)
        
        
        #unreal engine 5  3d realistic
        if args.classes == "person":
            names = ['person',"man","woman","child","boy","girl","old man","teenager"]
            h = choice(names)
            prompts = ["a {}, full body, no background".format(h)]
            
        elif args.classes == "tvmonitor":
            prompts = ["a tv, no background"]
            
        elif args.classes == "cat":
            prompts = ["a cat, full body, no background"]
            
        elif args.classes == "sheep":
            prompts = ["a sheep, full body, no background"]
            
        elif args.classes == "horse":
            prompts = ["a horse, full body, no background"]
            
        elif args.classes == "cow":
            prompts = ["a cow, full body, no background"]
            
        elif args.classes == "dog":
            prompts = ["a dog, full body, no background"]
            
        elif args.classes == "diningtable":
            prompts = ["a table, no background"]
            
        else:
            prompts = ["a {}, no background".format(args.classes)]
            

        print(image_cnt)
        print(prompts)
        
        
        controller = AttentionStore()
        image_cnt+=1
        image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu,out_put = os.path.join(image_path,"image_{}_{}.jpg".format(args.classes,image_cnt)),ldm_stable=ldm_stable)
        save_cross_attention(image[0].copy(),controller, res=32, from_where=("up", "down"),out_put = os.path.join(mask_path,"image_{}_{}.png".format(args.classes,image_cnt)),atten_put = os.path.join(atten_path,"image_{}_{}.png".format(args.classes,image_cnt)),image_cnt=image_cnt,class_one=args.classes,prompts=prompts,tokenizer=tokenizer)




    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", default="dog", type=str)
    parser.add_argument("--thread_num", default=8, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--image_number", default=None, type=str)
    args = parser.parse_args()
    
    args.output = os.path.join(args.output, "VOC_Multi_Attention_{}_sub_{}_NoClipRetrieval".format(args.classes,args.image_number))
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    result_dict = mp.Manager().dict()
    mp = mp.get_context("spawn")
    processes = []
#     per_thread_video_num = int(len(coco_category_list)/thread_num)

    print('Start Generation')
    for i in range(args.thread_num):
#         if i == thread_num - 1:
#             sub_video_list = coco_category_list[i * per_thread_video_num:]
#         else:
#             sub_video_list = coco_category_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]

        p = mp.Process(target=sub_processor, args=(i, args))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    



