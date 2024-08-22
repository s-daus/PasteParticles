import random
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from typing import Dict, Union
from utils.data_operations import get_image_files, split_dataset
from utils.image_operations import generate_mask
from os import PathLike


class DataLoader:
    """
    Loads a batch of images and their corresponding masks for synthetic image generation.

    Attributes:
        image_folder (Union[str, PathLike]): Directory containing images.
        mask_folder (Union[None, str, PathLike]): Directory containing masks corresponding to images.
        mask_areas (pd.DataFrame): DataFrame containing mask file names and their areas.
        target_size (tuple): Desired output size of the images.
        area_fraction (float): Target fraction of the target area to be covered by objects in the images.

    Methods:
        load_batch(): Loads a batch of images and their corresponding masks. Returns a tuple of images, binaries (masks),
                      and names of the selected objects.
    """

    def __init__(self, image_folder: Union[str, PathLike], mask_folder: Union[None, str, PathLike],
                 area_fraction: float, mask_areas: pd.DataFrame, target_size: tuple):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.mask_areas = mask_areas
        self.target_size = target_size
        self.area_fraction = area_fraction
        self.all_toy_files = get_image_files(self.image_folder)

    def load_batch(self) -> tuple:
        batch_image_files = self._get_batch_files()
        random.shuffle(batch_image_files)
        images = [self._load_image(file) for file in batch_image_files]
        if self.mask_folder:
            binaries = [self._load_image(file, is_mask=True) for file in batch_image_files]
        else:
            binaries = [generate_mask(img) for img in images]
        names = [f[:-4] for f in batch_image_files]
        return images, binaries, names

    def _get_batch_files(self) -> list:
        # Shuffle the dataframe
        total_area = self.target_size[0] * self.target_size[1]
        target_area = self.area_fraction * total_area
        shuffled_df = self.mask_areas.sample(frac=1).reset_index(drop=True)

        # Iterate and sum areas
        selected_objects = []
        accumulated_area = 0
        for _, row in shuffled_df.iterrows():
            accumulated_area += row['areas']
            selected_objects.append(Path(row['masks']).name)
            if accumulated_area >= target_area:
                break
        return selected_objects

    def _load_image(self, file, is_mask=False) -> np.ndarray:
        folder = self.mask_folder if is_mask else self.image_folder
        image = io.imread(Path(folder, file))
        return image


