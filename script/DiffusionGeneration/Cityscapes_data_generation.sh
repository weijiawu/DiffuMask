# generating Synthetic data and saving Attention Map
# /home/wangjue_Cloud/wuweijia  ./DiffSeg_Data
python3 ./Stable_Diffusion/parallel_generate_Cityscapes_Attention_AnyClass.py --classes person --thread_num 8 --output ./DiffMask_Cityscapes/ --image_number 100000