bug for cannot import name 'autocast' from 'torch'


# https://github.com/pesser/stable-diffusion/issues/14
For me, the issue was resolved by updating txt2img.py line 12 from from torch import autocast to from torch.cuda.amp import autocast.

# boat
误检（fp）有点多 63->74
解决：去掉clipretrieval prompt,加入更多的negative image

# aeroplane
更多数据量
或者更多的negative image


#
open world
cross validation (泛化性)

# VOC
aeroplane:  VOC_Multi_Attention_aeroplane_sub_30000_NoClipRetrieval   select_91.0.txt   1
bicycle:    VOC_Multi_Attention_bicycle_sub_50000_ClipRetrieval  select_85.0.txt        19 -> 2
bird:       VOC_Multi_Attention_bird_sub_30000_NoClipRetrieval   select_92.0.txt        3
boat        VOC_Multi_Attention_boat_sub_50000_ClipRetrieval     select_88.0.txt        4
bottle      VOC_Multi_Attention_bottle_sub_30000_ClipRetrieval   select_91.0.txt        5            （no 遮挡和仿射变换）
bus         VOC_Multi_Attention_bus_GPT3                         select_94.0.txt        6             (no sub class)
car         VOC_Multi_Attention_car_sub_retrieval_50000_GPT3     select_86.txt          7
cat         VOC_Multi_Attention_cat_sub_30000_ClipRetrieval      select_91.0.txt         8
chair       VOC_Multi_Attention_chair_sub_30000_NoClipRetrieval  select_88.0.txt        9
cow         VOC_Multi_Attention_cow_sub_30000_NoClipRetrieval    select_95.0.txt        10
diningtable VOC_Multi_Attention_diningtable_sub_30000_NoClipRetrieval select_80.0.txt   11
dog         VOC_Multi_Attention_dog_sub_30000_NoClipRetrieval    select_92.0.txt        12
horse       VOC_Multi_Attention_horse_sub_30000_NoClipRetrieval  select_90.0.txt        13
motorbike   VOC_Multi_Attention_motorbike_sub_50000_Clipretrieval select_94.0.txt       14
person      VOC_Multi_Attention_person_sub_20000_GPT3            select_75.0.txt        15
pottedplant VOC_Multi_Attention_pottedplant_sub_30000_NoClipRetrieval select_91.0.txt   16
sheep       VOC_Multi_Attention_sheep_sub_30000_NoClipRetrieval   select_91.0.txt       17
sofa        VOC_Multi_Attention_sofa_sub_15000_NoClipRetrieval    select_91.0.txt       18   (8x8 attention map)
train       VOC_Multi_Attention_train_sub_15000_NoClipRetrieval   select_88.0.txt       19   (8x8 attention map)
tvmonitor   VOC_Multi_Attention_tvmonitor_sub_20000_NoClipRetrieval select_85.0.txt     20