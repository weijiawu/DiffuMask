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


# % road          : 0.984      nan
# % sidewalk      : 0.868      nan
# % building      : 0.927      nan
# % wall          : 0.500      nan
# % fence         : 0.607      nan
# % pole          : 0.691      nan
# % traffic light : 0.660      nan
# % traffic sign  : 0.768      nan
# % vegetation    : 0.926      nan
# % terrain       : 0.650      nan
# % sky           : 0.949      nan
# % person        : 0.830    0.652
# % rider         : 0.592    0.348
# % car           : 0.945    0.875
# % truck         : 0.645    0.334
# % bus           : 0.919    0.641
# % train         : 0.811    0.541
# % motorcycle    : 0.676    0.352
# % bicycle       : 0.781    0.566
coco_category_list = [ 
    "road",'sidewalk','building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle']


VOC_category_list_check = {
    'road':['road'],
    'sidewalk':['sidewalk'],
    'building':['building'],
    'wall':['wall'],
    'fence':['fence'],
    'pole':['pole'],
    'traffic light':['traffic',"light"],
    'traffic sign':['traffic',"sign"],
    'vegetation':['vegetation'],
    'terrain':['terrain'],
    'sky':['sky'],
    'truck':['truck'],
    'rider':['rider'],
    'car':['car'],
    'person':coco_category_list_check_person,
    'bus':['bus'],
    'train':['train'],
    'motorcycle':['motorcycle'],
    'bicycle':['bicycle']
    }


coco_category_list_check = [    "road",'sidewalk','building',
    'wall',
    'fence',
    'pole',
    'light',
    'sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle']

human_verhicle_check = [
'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle'
]
# coco_category_to_id_v1 = {
#     0: 'road',
#     1: 'sidewalk',
#     2: 'building',
#     3: 'wall',
#     4: 'fence',
#     5: 'pole',
#     6: 'traffic light',
#     7: 'traffic sign',
#     8: 'vegetation',
#     9: 'terrain',
#     10: 'sky',
#     11: 'person',
#     12: 'rider',
#     13: 'car',
#     14: 'truck',
#     15: 'bus',
#     16: 'train',
#     17: 'motorcycle',
#     18: 'bicycle'
# }



coco_category_to_id_v1 = { 'road':0,
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
                        
def save_cross_attention(orignial_image,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,out_put="./test_1.jpg",image_cnt=0,class_one=None,prompts=None , tokenizer=None,mask_diff=None):
    
    
    orignial_image = orignial_image.copy()
    show = True
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts)
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
    
    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts)
    attention_maps_32 = attention_maps_32.sum(0) / attention_maps_32.shape[0]
    
    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select,prompts=prompts)
    attention_maps_64 = attention_maps_64.sum(0) / attention_maps_64.shape[0]
    

    cam_dict = {}
    for idx, class_one in enumerate(coco_category_list):
        
        gt_kernel_final = np.zeros((512,512), dtype='float32')
        number_gt = 0
        for i in range(len(tokens)):
            class_current = decoder(int(tokens[i])) 
            
            
            category_list_check = VOC_category_list_check[class_one]
            
            if class_current not in category_list_check:
                continue
#             print(class_current,category_list_check,class_one)
#             if class_one != "person":
#                 if class_current not in coco_category_list_check:
#                     continue
#             else:
#                 if class_current not in coco_category_list_check_person:
#                     continue
            
            
            image_16 = attention_maps[:, :, i]
            image_16 = cv2.resize(image_16.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_16 = image_16 / image_16.max()
            
            image_32 = attention_maps_32[:, :, i]
            image_32 = cv2.resize(image_32.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_32 = image_32 / image_32.max()
            
            image_64 = attention_maps_64[:, :, i]
            image_64 = cv2.resize(image_64.numpy(), (512, 512), interpolation=cv2.INTER_CUBIC)
            image_64 = image_64 / image_64.max()
            
            image = (image_16 + image_32 + image_64) / 3

            
            gt_kernel_final += image.copy()
            number_gt += 1

        if number_gt!=0:
            gt_kernel_final = gt_kernel_final/number_gt
        
        id_ = coco_category_to_id_v1[class_one]
        cam_dict[id_] = gt_kernel_final
#         print(id_,gt_kernel_final.sum(),number_gt)
        
#         for i in range(19):
#             if i == id_:
#                 cam_dict[i] = gt_kernel_final
#             else:
#                 image_h, image_w = gt_kernel_final.shape[0:2]
#                 gt_kernel = np.zeros((image_h,image_w), dtype='uint8')
#                 cam_dict[i] = gt_kernel

    np.save(out_put, cam_dict)

    

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
    
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=200, generator=generator, low_resource=LOW_RESOURCE)

    
    ptp_utils.view_images(images_here,out_put = out_put)
    return images_here, x_t


def clipretrieval(text,check):
    sensitive_word = ["vector","stock","3d","-3d","-","blur","Vector","blurred","shot","close-up","Headlight","Stock","headlights","Defocused","Close-up","3D","cartoon","interior","internal"] 
    prompts = []
    for prompt in tqdm(text):
        try:
            client = ClipClient(
                url="https://knn5.laion.ai/knn-service",
                indice_name="laion5B",
                aesthetic_score=9,
                aesthetic_weight=0.5,
                modality=Modality.IMAGE,
                num_images=3000,
            )
            results = client.query(text=prompt)
        except:
            client = ClipClient(
                url="https://knn5.laion.ai/knn-service",
                indice_name="laion5B",
                aesthetic_score=9,
                aesthetic_weight=0.5,
                modality=Modality.IMAGE,
                num_images=1000,
            )
            results = client.query(text=prompt)
            
        for i,line in enumerate(results):
            caption = line["caption"]
            caption_split = caption.split(" ")
            continue_flag = False
            
            # filter noise from other classes
            for cls in human_verhicle_check:
                if cls not in check and cls in caption_split:
                    continue_flag = True
            if continue_flag:
                continue
                
                
            continue_flag = True
            for chec in check:
                sen_flag = True
                for c in sensitive_word:
                    if c in caption_split:
                        sen_flag = False
                
                
                        
                    
                if chec in caption_split[:5] and sen_flag:
                    continue_flag=False
                    
            if continue_flag:
                continue
                
            if len(caption_split)>50:
                continue
#             if ""
            prompts.append("Photo of "+caption)
    
    return prompts

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
    npy_path = os.path.join(args.output,"npy")
    
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    
#     prompts_list = ["Photo of a {}".format(args.classes),
#                    "a {}".format(args.classes),
#                    "{}".format(args.classes)]
    
    # bird         https://byjus.com/english/birds-name/
    bird_sub_classes = ["Masai Ostrich","Macaw","Eagle","Duck","Hen","Parrot","Peacock","Dove","Stork","Swan","Pigeon","Goose",
                        "Pelican","Macaw","Parakeet","Finches","Crow","Raven","Vulture","Hawk","Crane","Penguin", "Hummingbird",
                        "Sparrow","Woodpecker","Hornbill","Owl","Myna","Cuckoo","Turkey","Quail","Ostrich","Emu","Cockatiel"
                        ,"Kingfisher","Kite","Cockatoo","Nightingale","Blue jay","Magpie","Goldfinch","Robin","Swallow",
                        "Starling","Pheasant","Toucan","Canary","Seagull","Heron","Potoo","Bush warbler","Barn swallow",
                        "Cassowary","Mallard","Common swift","Falcon","Megapode","Spoonbill","Ospreys","Coot","Rail",
                        "Budgerigar","Wren","Lark","Sandpiper","Arctic tern","Lovebird","Conure","Rallidae","Bee-eater",
                        "Grebe","Guinea fowl","Passerine","Albatross","Moa","Kiwi","Nightjar","Oilbird","Gannet","Thrush",
                        "Avocet","Catbird","Bluebird","Roadrunner","Dunnock","Northern cardinal","Teal",
                        "Northern shoveler","Gadwall","Northern pintail",
                        "Hoatzin","Kestrel","Oriole","Partridge","Tailorbird","Wagtail","Weaverbird","Skylark"]
    
    # cat         https://www.purina.com/cats/cat-breeds?page=0
    cat_sub_classes = ["Abyssinian","American Bobtail","American Curl","American Shorthair","American Wirehair",
                       "Balinese-Javanese","Bengal","Birman","Bombay","British Shorthair","Burmese","Chartreux Cat",
                        "Cornish Rex","Devon Rex","Egyptian Mau","European Burmese","Exotic Shorthair","Havana Brown",
                       "Himalayan","Japanese Bobtail","Korat","LaPerm", "Maine Coon","Manx","Munchkin",
                       "Norwegian Forest","Ocicat","Oriental","Persian","Peterbald","Pixiebob","Ragamuffin",
                       "Ragdoll","Russian Blue","Savannah","Scottish Fold","Selkirk Rex","Siamese","Siberian",
                       "Singapura","Somali","Sphynx","Tonkinese","Toyger","Turkish Angora","Turkish Van",""]
    
    
    # bus         https://simple.wikipedia.org/wiki/Types_of_buses
    bus_sub_classes = ["Coach","Motor","School","Shuttle","Mini",
                       "Minicoach","Double-decker","Single-decker","Low-floor","Step-entrance","Trolley",
                        "Articulated","Guided","Neighbourhood","Gyrobus","Hybrid","Police",
                       "Open top","Electric","Transit","Tour","Commuter","Party",""]
    more_prompt_for_bus = ["Photo of a bus in the street", "Photo of a bus in the road",   "Photo of a bus in the street at night",      "Photo of the back of a bus", "Photo of the overturn of a bus", "Photo of a bus", "Photo of a bus", "Photo of a bus", "Photo of a bus", "Photo of a bus", "Photo of a bus"]
    
    # truck
    prompt_for_truck = ["Photo of a truck in the street", "Photo of a truck", "Photo of a truck in the road",   "Photo of a truck in the street at night",      "Photo of the back of a truck", "Photo of the overturn of a truck"]
    
    # bicycle
    prompt_for_bicycle = ["Photo of a bicycle in the street", "Photo of a bicycle", "Photo of a bicycle in the road",   "Photo of a bicycle in the street at night",      "Photo of the back of a bicycle"]
    
    # traffic light
    prompt_for_traffic_light = ["Photo of a traffic light in the street", "Photo of a traffic light", "Photo of a traffic light in the road",   "Photo of a traffic light in the street at night"]
    
    # motorcycle         https://ride.vision/blog/13-motorcycle-types-and-how-to-choose-one/
    motorcycle_sub_classes = ["Cruisers","Sportbikes","Standard & Naked ","Adventure","Dual Sports & Enduros",
                       "Dirtbikes","Electric","Choppers","Touring","Sport Touring","Vintage & Customs",
                        "Modern Classics","Commuters & Minis","Scooters",""]
    more_prompt_for_motorcycle = ["Photo of a motorcycle in the street", "Photo of a motorcycle in the road",   "Photo of a motorcycle in the street at night",      "Photo of the back of a motorcycle", "Photo of the overturn of a motorcycle"]
    
    
    # car         https://www.caranddriver.com/shopping-advice/g26100588/car-types/
    car_sub_classes = ["SEDAN","COUPE","SPORTS","STATION WAGON","HATCHBACK",
                       "CONVERTIBLE","SPORT-UTILITY VEHICLE","MINIVAN","PICKUP TRUCK","IT DOESN'T STOP THERE",""]
    more_prompt_for_car = ["Photo of a car in the street", "Photo of a car in the road",   "Photo of a car in the street at night",      "Photo of the back of a car", "Photo of the overturn of a car"]
#     car_sub_classes_2 = ["photo of a car"]
#     "photo of a car in street"
    
    # road
    road_sub_classes = ["Photo of street road in city", "Photo of street road at night in city"]
    
    
    
    map_dict = {"bird":bird_sub_classes,"cat":cat_sub_classes,"bus":bus_sub_classes}
    
    sub_classes = ["bird","cat"]
    if args.classes in sub_classes:
        prompts_list = ["Photo of a {} {}",
                       "a {} {}",
                       "{} {}"]
    
    elif args.classes == "road":
        road_sub_classes = clipretrieval(road_sub_classes,["road"])
        
    elif args.classes == "bus":
#         bus_sub_classes = ["Photo of a {} bus".format(name) for name in bus_sub_classes]
#         bus_sub_classes = bus_sub_classes + more_prompt_for_bus
        bus_sub_classes = more_prompt_for_bus
#         bus_sub_classes = clipretrieval(bus_sub_classes,["bus"])
    
    elif args.classes == "motorcycle":
        motorcycle_sub_classes = ["Photo of a {} motorcycle".format(name) for name in motorcycle_sub_classes]
        motorcycle_sub_classes = motorcycle_sub_classes + more_prompt_for_motorcycle
        motorcycle_sub_classes = clipretrieval(motorcycle_sub_classes,["motorcycle"])
        
    elif args.classes == "truck":
        truck_sub_classes = prompt_for_truck
        truck_sub_classes = clipretrieval(truck_sub_classes,["truck"])
    
    elif args.classes == "trafficlight":
        traffic_light_sub_classes = prompt_for_traffic_light
        traffic_light_sub_classes = clipretrieval(traffic_light_sub_classes,["traffic","light"])
        
    elif args.classes == "bicycle":
        bicycle_sub_classes = prompt_for_bicycle
#         truck_sub_classes = clipretrieval(truck_sub_classes,["truck"])
        
    elif args.classes == "car":
        car_sub_classes = ["Photo of a {} car".format(name) for name in car_sub_classes]
        car_sub_classes = car_sub_classes + more_prompt_for_car
        car_sub_classes = clipretrieval(car_sub_classes,["car"])
#         car_sub_classes = car_sub_classes_2
        
    elif args.classes == "person":
#         name = choicek
        names = ['person',"man","woman","child","boy","girl","old man","teenager"]
    
        prompts_list = ["photo of a {} is walking",
                        "photo of a {} is eating",
                        "photo of a {} is riding motorbike",
                        "photo of a {} is riding bicycle",
                        "photo of a {} is riding horse",
                        "photo of a {} is driving",  
                        "photo of a {} is play",  
                       
                        
                        
                        #sports
                        "photo of a {} is playing baseball","photo of a {} is playing Basketball", 
                        "photo of a {} is playing Badminton","photo of a {} is Swimming",
                        "photo of a {} is playing Bodybuilding","photo of a {} is playing Bowling",
                        "photo of a {} is dancing","photo of a {} is playing Football",
                        "photo of a {} is playing Golf","photo of a {} is playing Frisbee",
                        "photo of a {} is Skiing", "photo of a {} is playing Table Tennis",
                        "photo of a {} is doing Yoga","photo of a {} is doing Fitness",
                        "photo of a {} is doing Rugby","photo of a {} is doing Wrestling",
                        "photo of a {} is doing High jumping","photo of a {} is Cycling",
                        "photo of a {} is running","photo of a {} is Fishing",
                        "photo of a {} is doing Judo","photo of a {} is Climbing",
                        
                        # scenario
                        "photo of a {} is walking in the street","photo of a {} is in the road",
                        "photo of a {} is playing at home","photo of a {} is in the shopping center",
                        "photo of a {} is in on the mountain","photo of a {} is in on the mountain",
                        "photo of a {} is crossing a road","photo of a {} is sitting",
                        "photo of a {} is sitting at home", "photo of a {} is playing at sofa",
                        "photo of a {} is playing at home","photo of back of a {}",
                        
                        
                        #others  
                        "photo of a {} is cooking",
                       "photo of a {}",
                       "photo of arm of a {}",
                        "photo of foot of a {}",
                       "photo of a {} is running"]
        
        person_sub_classes = []
        for name in names:
            for prompts_line in prompts_list:
                person_sub_classes.append(prompts_line.format(name))
        
        print("prompt number:",len(car_sub_classes))
        person_sub_classes = clipretrieval(person_sub_classes,coco_category_list_check_person)
        
        
    else:
        prompts_list = ["Photo of a {}".format(args.classes),
                   "a {}".format(args.classes),
                   "{}".format(args.classes)]
    
    for rand in range(number_per_thread_num):
        g_cpu = torch.Generator().manual_seed(image_cnt)
#         prompts = [GPT3(args).split(".")[0].split(",")[0].replace("\n","")]
        
#         if args.classes not in prompts[0]:
#             continue
        
#         if random.random() >0.5:
        
        if args.classes in sub_classes:
            sub_cls = map_dict[args.classes]
            sub_class = choice(sub_cls)
            prompts = [choice(prompts_list).format(sub_class,args.classes)]
        elif args.classes == "car":
            prompts = [choice(car_sub_classes)]
        elif args.classes == "bus":
            prompts = [choice(bus_sub_classes)]
        elif args.classes == "motorcycle":
            prompts = [choice(motorcycle_sub_classes)]
        elif args.classes == "person":
            prompts = [choice(person_sub_classes)]
        elif args.classes == "road":
            prompts = [choice(road_sub_classes)]
        elif args.classes == "truck":
            prompts = [choice(truck_sub_classes)]
        elif args.classes == "bicycle":
            prompts = [choice(bicycle_sub_classes)]
        elif args.classes == "trafficlight":
            prompts = [choice(traffic_light_sub_classes)]    
            
            
        print(prompts)
        
        
        controller = AttentionStore()
        image_cnt+=1
        image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu,out_put = os.path.join(image_path,"image_{}_{}.jpg".format(args.classes,image_cnt)),ldm_stable=ldm_stable)
        save_cross_attention(image[0].copy(),controller, res=32, from_where=("up", "down"),out_put = os.path.join(npy_path,"image_{}_{}".format(args.classes,image_cnt)),image_cnt=image_cnt,class_one=args.classes,prompts=prompts,tokenizer=tokenizer)




    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", default="dog", type=str)
    parser.add_argument("--thread_num", default=8, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--image_number", default=None, type=str)
    args = parser.parse_args()
    
    args.output = os.path.join(args.output, "COCO2017_Multi_Attention_{}_sub_{}_Clipretrieval".format(args.classes,args.image_number))
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
    



