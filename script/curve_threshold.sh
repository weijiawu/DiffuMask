# run aff_prepare.py
# voc cityscapes
python curve_threshold.py --voc12_root voc12 --cam_dir ./DiffMask_VOC/VOC_Multi_Attention_bird_sub_15000_NoClipRetrieval/npy  --out_crf ./DiffMask_VOC/VOC_Multi_Attention_bird_sub_15000_NoClipRetrieval/refine_gt_crf --infer_list DiffMask_VOC/VOC_Multi_Attention_bird_sub_15000_NoClipRetrieval/train_image --dataset voc