# generating Synthetic data and saving Attention Map
# /home/wangjue_Cloud/wuweijia  ./DiffSeg_Data
# python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes bird --thread_num 8 --output ./DiffMask_VOC/ --image_number 15000

python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes aeroplane --thread_num 8 --output ./DiffMask_VOC/ --image_number 8000