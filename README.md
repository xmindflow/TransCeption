# TransCeption: Hierarchical and Inception-like Transformer Design for Medical Image Segmentation

---

TransCeption is a U-shaped hierarchical architecture which aggregates the inception-like structure in the encoder based on the pure transformer network.

 In this approach,  

- We employ ResInception Patch Merging (RIPM) and Multi-branch Transformer Block (MB transformer) at each stage of the encoder to extract the multi-scale features of both global contextual information and local finer details.

- Intra-stage Feature Fusion Module (IFF) is introduced to emphasize the interactions across the channel dimension of concatenated feature maps with the crucial positional information properly retained.

- We redesign Dual Transformer Bridge based on the Enhanced Transformer Context Bridge \cite{huang2021missformer} to further model inter-stage correlations of hierarchical multi-scale features.
  

---

## Updates

We thank the great work of MBTransformer, Swin-Unet and MISSFormer.

## Requirements

This code is implemented in python 3.6.3 using PyTorch library 1.8.0 and tested in ubuntu OS. We use the libraries of these versions:

- Python 3.6

- Torch 1.8.0

- torchvision 0.2.2

- numpy 1.19.5

To set up the correct environment, we recommend running the following code to install the requirements.

```python
pip install -r requirements.txt
```

## Dataset preparation

- Synapse Dataset: please go to "./datasets/README.md" for the details about preparing preprocessed Synapse dataset. Or download the Synapse Dataset from [here](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi) (??)

- ISIC 2018: please download the ISIC 2018 train dataset from [here](??) (??)

## Train

Run the following code to train TransCeption on the Synapse Dataset:



```python
--dataset Synapse --base_lr 0.05 --max_epochs 500 --eval_interval 20 --model_name TransCeption --batch_size 16 --root_path <your path to ./Synapse/train_npz> --output_dir <your output path>
```

## Test

python test.py --dataset Synapse --base_lr 0.05 --model_name TransCeption --output_dir <your output path> --br_config 2 --weight_pth <your path to .pth file>

```python```  
python test.py --dataset Synapse --base_lr 0.05 --model_name TransCeption --output_dir <your output path> --br_config 2 --weight_pth <your path to .pth file>
```

## Quick Overview

![](assets/519bc17861600bf55465c88b38ac52105071d6ea.png)

## Results