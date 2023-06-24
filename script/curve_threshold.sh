# run aff_prepare.py
# voc cityscapes
python curve_threshold.py --voc12_root voc12 --cam_dir ./DiffMask_VOC/VOC_Multi_Attention_aeroplane_sub_8000_NoClipRetrieval_sample/npy  --out_crf ./DiffMask_VOC/VOC_Multi_Attention_aeroplane_sub_8000_NoClipRetrieval_sample/refine_gt_crf --infer_list DiffMask_VOC/VOC_Multi_Attention_aeroplane_sub_8000_NoClipRetrieval_sample/train_image --dataset voc