# run aff_prepare.py
# dataset voc12  Cityscapes
# class_number VOC:21  Cityscapes:20(add a background class)
python aff_prepare.py --dataset VOC --cam_dir ./DiffMask_VOC/VOC_Multi_Attention_aeroplane_sub_8000_NoClipRetrieval_sample/npy  --class_number 21 --out_crf ./DiffMask_VOC/VOC_Multi_Attention_aeroplane_sub_8000_NoClipRetrieval_sample/crf_alpha_dir --infer_list DiffMask_VOC/VOC_Multi_Attention_aeroplane_sub_8000_NoClipRetrieval_sample/train_image --classes aeroplane 