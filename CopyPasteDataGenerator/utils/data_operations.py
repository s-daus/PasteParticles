import pathlib
from PIL import Image
import numpy as np


def is_legible_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False


def get_image_files(directory):
    path = pathlib.Path(directory)
    image_files = []

    for file_path in path.glob('**/*'):
        if file_path.is_file() and is_legible_image_file(file_path):
            image_files.append(file_path)
    return image_files


def split_dataset(number: int, sizes: list):
    total_size = sum(sizes)

    # Check if the sizes add up to the number
    if np.isclose(total_size, 1) is False:
        raise ValueError("sizes do not add up to  1")

    # Calculate the size of each chunk
    chunk_sizes = [int(size * number) for size in sizes]

    # Calculate any remaining size and distribute it evenly among the chunks
    remaining_size = number - sum(chunk_sizes)
    for i in range(remaining_size):
        chunk_sizes[i % len(chunk_sizes)] += 1

    return chunk_sizes


def create_target_masks(instance_masks):
    targets = np.array([np.max(i) for i in instance_masks])
    instance_masks = np.array(instance_masks)
    instance_masks[instance_masks != 0] = 1
    return instance_masks, targets


def find_matching_key(my_dict, my_string):
    for key in my_dict:
        if key in my_string:
            return key
    return "key doesn't exist"


def get_key_from_value(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key




