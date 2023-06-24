# generating Synthetic data and saving Attention Map
# python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes bird --thread_num 8 --output ./DiffMask_VOC/ --image_number 15000

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes aeroplane --thread_num 4 --output ./DiffMask_VOC/ --image_number 8000 --MY_TOKEN 'your key'
