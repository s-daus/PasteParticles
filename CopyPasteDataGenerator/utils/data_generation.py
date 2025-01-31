import concurrent.futures
import numpy as np
from pathlib import Path

import pandas as pd

from utils.data_loader import DataLoader
from utils.image_creator_class import ArtificialImageCreator
from utils.data_operations import create_target_masks, get_image_files
from utils.plotting import plot_instance, plot_semantic
from skimage import io
from tqdm import tqdm


def generate_data(data: dict, data_type: str, number_of_cores: int = 1):
    """
    Generates synthetic images and masks based on specified settings.

    Args:
        data (dict): Configuration data for image generation.
        data_type (str): Type of data to generate ('instance' or 'semantic').
        number_of_cores (int, optional): Number of processing cores to use. Defaults to 1.

    Returns:
        np.ndarray: Array of indices of the generated images.
    """
    output_dir = Path(data['save_path'])
    image_dir = Path(output_dir, 'images')
    mask_dir = Path(output_dir, 'masks')
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    mask_areas = pd.read_csv(data['object_area_dataframe_path'], usecols=['masks', 'areas'])

    if data_type not in ['instance', 'semantic']:
        raise ValueError('data_type must be either "instance" or "semantic".')

    with concurrent.futures.ProcessPoolExecutor(number_of_cores) as executor:
        futures = []
        i_0 = len(get_image_files(image_dir))
        for i in range(i_0, i_0 + data['number_images']):
            futures.append((i, executor.submit(generate_image_and_mask, data, mask_areas, data_type == 'instance')))

        # Create a tqdm progress bar
        pbar = tqdm(total=len(futures), desc="Processing images", dynamic_ncols=True)

        for i, future in futures:
            image, mask = future.result()
            save_func = save_instance if data_type == 'instance' else save_semantic
            save_func(image, mask, i, image_dir, mask_dir, data)

            # Update the progress bar
            pbar.update()

        # Close the progress bar when done
        pbar.close()
        return np.arange(i_0, i_0 + data['number_images'])


def plot_images(data: dict, data_type: str, indexes):
    output_dir = Path(data['save_path'])
    image_dir = Path(output_dir, 'images')
    mask_dir = Path(output_dir, 'masks')

    plot_func = plot_instance if data_type == 'instance' else plot_semantic
    for i in indexes:
        image_path = image_dir / f"{i}{data['save_extension']}"
        mask_path = mask_dir / f"{i}" if data_type == 'instance' else mask_dir / f"{i}{data['save_extension']}"
        if image_path.exists() and mask_path.exists():
            plot_func(image_path, mask_path, data['class_colors'])


def generate_image_and_mask(data: dict, mask_areas: pd.DataFrame, is_instance=True) -> tuple:
    aic = ArtificialImageCreator(class_colors=data['class_colors'], augmentation=data['augmentation'],
                                 path_bg=data['path_bg'], max_overlap=data['max_overlap'], update_overlapping_masks=data['update_overlapping_masks'])
    data_base = DataLoader(data['path_toys'], data['path_masks'], mask_areas=mask_areas,
                           area_fraction=data['area_fraction'], target_size=data["target_imagesize"])
    batch = data_base.load_batch()
    aic.create_synthetic_image(batch)
    image = aic.Toy
    return image, aic.InstanceMasks if is_instance else aic.Label


def save_instance(image, instance_masks, i, image_dir, mask_dir, data):
    masks, targets = create_target_masks(instance_masks)
    mask_path = Path(mask_dir, str(i))
    mask_path.mkdir(parents=True, exist_ok=True)
    io.imsave(image_dir / f"{i}{data['save_extension']}", image, check_contrast=False)
    for j in range(masks.shape[0]):
        io.imsave(mask_path / f"{j}{data['save_extension']}", masks[j, :, :], check_contrast=False)
    np.save(mask_path / f"{i}", targets)


def save_semantic(image, label, i, image_dir, mask_dir, data):
    io.imsave(image_dir / f"{i}{data['save_extension']}", image, check_contrast=False)
    io.imsave(mask_dir / f"{i}{data['save_extension']}", label, check_contrast=False)
