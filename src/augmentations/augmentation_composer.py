from torchvision import transforms
from yucca.data.augmentation.transforms.formatting import (
    AddBatchDimension,
    RemoveBatchDimension,
)
from yucca.data.augmentation.transforms.BiasField import BiasField
from yucca.data.augmentation.transforms.Blur import Blur
from yucca.data.augmentation.transforms.copy_image_to_label import CopyImageToLabel
from yucca.data.augmentation.transforms.Gamma import Gamma
from yucca.data.augmentation.transforms.Ghosting import MotionGhosting
from yucca.data.augmentation.transforms.Noise import (
    AdditiveNoise,
    MultiplicativeNoise,
)
from yucca.data.augmentation.transforms.Ringing import GibbsRinging
from yucca.data.augmentation.transforms.SimulateLowres import SimulateLowres
from yucca.data.augmentation.transforms.Spatial import Spatial


def get_pretrain_augmentations(patch_size, preset):
    assert preset in ["none", "spatial", "all"]

    if preset == "none":
        augmentations = [CopyImageToLabel(copy=True)]

    elif preset == "spatial":
        augmentations = [spatial_augmentation(patch_size), CopyImageToLabel(copy=True)]

    elif preset == "all":
        augmentations = [
            spatial_augmentation(patch_size),
            CopyImageToLabel(copy=True),
        ] + intensity_augmentations()

    return transforms.Compose([AddBatchDimension()] + augmentations + [RemoveBatchDimension()])


def get_val_augmentations():
    return transforms.Compose([AddBatchDimension(), CopyImageToLabel(copy=True), RemoveBatchDimension()])


def get_finetune_augmentations(patch_size, preset):
    assert preset in ["none", "spatial", "all"]

    if preset == "none":
        return None

    elif preset == "spatial":
        augmentations = [spatial_augmentation(patch_size)]

    elif preset == "all":
        augmentations = [spatial_augmentation(patch_size)] + intensity_augmentations()

    return transforms.Compose([AddBatchDimension()] + augmentations + [RemoveBatchDimension()])


def spatial_augmentation(patch_size):
    return Spatial(
        patch_size=patch_size,
        crop=True,
        random_crop=False,
        cval="min",
        p_deform_per_sample=0.33,
        deform_sigma=(20, 30),
        deform_alpha=(200, 600),
        p_rot_per_sample=0.2,
        p_rot_per_axis=0.66,
        x_rot_in_degrees=(-30.0, 30.0),
        y_rot_in_degrees=(-30.0, 30.0),
        z_rot_in_degrees=(-30.0, 30.0),
        p_scale_per_sample=0.2,
        scale_factor=(0.9, 1.1),
        skip_label=True,
        clip_to_input_range=True,
    )


def intensity_augmentations():
    return [
        AdditiveNoise(p_per_sample=0.2, mean=(0.0, 0.0), sigma=(1e-3, 1e-4), clip_to_input_range=True),
        Blur(p_per_sample=0.2, p_per_channel=0.5, sigma=(0.0, 1.0), clip_to_input_range=True),
        MultiplicativeNoise(p_per_sample=0.2, mean=(0, 0), sigma=(1e-3, 1e-4), clip_to_input_range=True),
        MotionGhosting(p_per_sample=0.2, alpha=(0.85, 0.95), num_reps=(2, 11), axes=(0, 3), clip_to_input_range=True),
        GibbsRinging(p_per_sample=0.2, cut_freq=(96, 129), axes=(0, 3), clip_to_input_range=True),
        SimulateLowres(p_per_sample=0.2, p_per_channel=0.5, p_per_axis=0.33, zoom_range=(0.5, 1.0), clip_to_input_range=True),
        BiasField(p_per_sample=0.33, clip_to_input_range=True),
        Gamma(p_per_sample=0.2, p_invert_image=0.05, gamma_range=(0.5, 2.0), clip_to_input_range=True),
    ]
