# DiffuMask
DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models


## Getting Started with DiffuMask
### Conda env installation

```
 install pydensecrf https://github.com/lucasb-eyer/pydensecrf
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

pip install -r requirements.txt
```

### 1. Data and mask generation
```
# generating data and attention map witn stable diffusion
sh ./script/DiffusionGeneration/VOC_data_generation.sh
```

### 2. Refine Mask with AffinityNet (Coarse Mask)
```
# prepare training data for affinity net
sh ./script/prepare_aff_data.sh

# train affinity net
sh ./script/train_affinity.sh

# inference affinity net
sh ./script/infer_aff.sh

# generate accurate pseudo label with CRF
sh ./script/curve_threshold.sh
```

### 3. Noise Learning (Cross Validation)


### 4. Training Mask2former with clear data