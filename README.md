# AMAES

Official Pytorch implementation of AMAES from the paper

> **[AMAES: Augmented Masked Autoencoder Pretraining on Public Brain MRI Data for 3D-Native Segmentation](https://arxiv.org/abs/2408.00640v2)** <br>
> ADSMI @ MICCAI 2024 <br>
> [Asbjørn Munk*](https://asbn.dk), [Jakob Ambsdorf*](https://scholar.google.de/citations?user=Cj2NnUIAAAAJ&hl=en), [Sebastian Llambias](https://scholar.google.com/citations?user=axb26RQAAAAJ&hl=en), [Mads Nielsen](https://scholar.google.de/citations?user=2QCJXEkAAAAJ&hl=en)
>
> Pioneer Centre for AI & University of Copenhagen
>
> \* Equal Contribution

_Efficient_ pretraining for 3D segmentation models using MAE and augmentation reversal on a large domain-specific dataset.

**Overview**
![results-no_arrow](https://github.com/user-attachments/assets/9e125d38-34d9-48da-83c8-20db612fb153)

**Method**
![abstract](https://github.com/user-attachments/assets/37518818-2b7d-415e-86bb-47ded4e41545)



# 🧠BRAINS-45K dataset
All models are pretrained on 🧠BRAINS-45K, the largest pretraining dataset available for brain MRI.

**All code necesarry to reproduce the dataset will be made available as soon as possible**.

# Model checkpoints

All checkpoints have been pretrained on 🧠BRAINS-45K for _100_ epochs using AMAES.

| Model     | Parameters      | Checkpoint |   |        |
|-----------|-----------------|------------|---|--------|
|           | _M_             | Zenodo     | 🤗 | Kaggle |
| U-Net XL  | 90              | [Download](https://zenodo.org/records/13604788/files/unet_xl_lw_dec_fullaug.pth?download=1) |   |        |
| U-Net B   | 22              | [Download](https://zenodo.org/records/13604788/files/unet_b_lw_dec_fullaug.pth?download=1) |   |        |
| MedNeXt-L | 55              | [Download](https://zenodo.org/records/13604788/files/mednext_l3_lw_dec_fullaug.pth?download=1) |   |        |
| MedNeXt-M | 21              | [Download](https://zenodo.org/records/13604788/files/mednext_m3_lw_dec_fullaug.pth?download=1) |   |        |

All models were pretrained on 2xH100 GPUs with 80GB of memory.

# Running the code

1. Install [Poetry](https://python-poetry.org/docs/).
2. Create environment by calling `poetry install`.

## Setup data
AMAES is using the Yucca library for handling 3D medical data.

Guide on how to setup data comming soon.

## Pretraining

To pretrain using AMAES run

```
poetry run src/pretrain.py --base_path=<path to base data directory>
```

## Finetuning

To finetune using AMAES, run

```
poetry run src/train.py --base_path=<path to base data directory> --pretrained_weights_path="<path_to_checkpoint>" --model=<model_to_instantiate>
```
Note that the checkpoint must match the model provided. For instance, to finetune `unet_xl_lw_dec_fullaug.pth` use `--model=unet_xl`.


# Citation

Please use

```
@article{munk2024amaes,
  title={AMAES: Augmented Masked Autoencoder Pretraining on Public Brain MRI Data for 3D-Native Segmentation},
  author={Munk, Asbjørn and Ambsdorf, Jakob and Llambias, Sebastian and Nielsen, Mads},
  journal={arXiv preprint arXiv:2408.00640},
  year={2024}
}
```
