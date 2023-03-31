# VOC
python evaluation.py --predict_dir ./DiffMask_VOC/VOC_Multi_Attention_bird_sub_15000_NoClipRetrieval/refine_gt_crf --gt_dir ./DiffMask_VOC/VOC_Multi_Attention_bird_sub_15000_NoClipRetrieval/cross_validation --output_dir ./DiffMask_VOC/VOC_Multi_Attention_bird_sub_15000_NoClipRetrieval --dataset VOC --type png --class bird --threshold 0.7 --select True 

# --select True  
# --curve True  --select False  

# Cityscapes
# python evaluation.py --predict_dir ./DiffMask_Cityscapes/Cityscapes_Multi_Attention_person_sub_100000_Clipretrieval/refine_gt_crf --gt_dir ./DiffMask_Cityscapes/Cityscapes_Multi_Attention_person_sub_100000_Clipretrieval/cross_validation --output_dir ./DiffMask_Cityscapes/Cityscapes_Multi_Attention_person_sub_100000_Clipretrieval --dataset Cityscapes --type png --class person --threshold 75 --select True 

#debug Cityscapes
# python evaluation.py --predict_dir /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/demo/cityscapes_pred --gt_dir /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/datasets/cityscapes/gtFine/val --type png --class bicycle --dataset Cityscapes
# --select True

#debug VOC
# python evaluation.py --predict_dir /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/demo/pred --gt_dir /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/datasets/PascalVOC12/SegmentationClassAug --type png --class sheep --select True

#debug COCO2017
# python evaluation.py --predict_dir /mmu-ocr/pub/weijiawu/Code/Diffusion_Model/Mask2Former/demo/COCO2017_pred --gt_dir /mmu-ocr/pub/weijiawu/Code/Diffusion_Model/Mask2Former/datasets/COCO2017/SegmentationClass/panoptic_semseg_val2017 --type png --class person --dataset COCO2017 --select True
#
#



## mask  ground_truth  ground_truth_fake   


#debug ADE20K
# python DiffMask_VOC/evaluation.py --predict_dir /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/demo/ADE_pred --gt_dir /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/datasets/ADEChallengeData2016/annotations_detectron2/validation --type png --select True --class car