# DiffuMask
DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models

<p align="center">
<img src="./1684329483514.jpg" width="800px"/>  
<br>
</p>


## :hammer_and_wrench: Getting Started with DiffuMask
### Conda env installation

```sh
conda create -n DiffuMask python=3.8

conda activate DiffuMask
```

```
 install pydensecrf https://github.com/lucasb-eyer/pydensecrf
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

pip install -r requirements.txt
```
```
If there is an error: 

bug for cannot import name 'autocast' from 'torch', 

please refer to the website:  

https://github.com/pesser/stable-diffusion/issues/14
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