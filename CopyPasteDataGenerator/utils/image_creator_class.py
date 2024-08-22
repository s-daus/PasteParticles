import cv2
import numpy as np
from os import PathLike
from typing import Union
from skimage import io
import random
from utils.plotting import plot_instance_inline
from utils.image_operations import augmentations, random_objects, flip_image, create_blending_mask, subtract_background, \
    adjust_object_brightness, bokeh_blur
from utils.data_operations import get_image_files, find_matching_key


class ArtificialImageCreator:
    """
    Creates synthetic images by placing objects on various backgrounds with specified augmentations.

    Attributes:
        class_colors (dict): Mapping of class names to their corresponding colors in the output mask.
        update_overlapping_masks (dict, optional): Dictionary specifying for each class if occluded region should be cut from ground truth for object overlay.
        augmentation (dict, optional): Dictionary specifying augmentation operations and their probabilities.
        path_bg (Union[str, PathLike], optional): Path to the background images directory.
        max_overlap (float): Maximum allowed overlap ratio between objects in the image.
        testing_mode (bool): if testing mode is chosen, no blending of objects will be done

    Methods:
        create_synthetic_image(objects): Places given objects on a background to create a synthetic image.
        read_bg(): Reads a random background image from the specified directory or creates a plain background.
    """

    def __init__(self, class_colors: dict, update_overlapping_masks: dict, augmentation: dict = None,
                 path_bg: Union[str, PathLike] = None,
                 max_overlap: float = 0.0, testing_mode: bool = False):
        self.class_colors = class_colors
        self.augmentation = augmentation
        self.bg_folder = path_bg
        self.Toy = None
        self.Label = None
        self.Background = None
        self.max_overlap = max_overlap
        self.instance_areas = {}
        self.InstanceMasks = []
        self.update_overlapping_masks = update_overlapping_masks
        self.adjusted_objects = []
        self.testing_mode = testing_mode

    def create_synthetic_image(self, objects):
        Background = self.read_bg(shape=objects[0][0].shape)
        self.Background = Background
        self.Toy = self.Background.copy()
        self.Label = np.zeros(self.Background.shape[:2], dtype='uint8')

        self._place_distractors(objects)
        self._position_objects(objects, blending=not self.testing_mode)

    def _place_distractors(self, objects):
        # get number of distractors
        min_distractor = int(0.2 * len(objects[0]))
        max_distractor = int(0.6 * len(objects[0]))
        num_distractor = np.random.randint(min_distractor, max_distractor, 1)

        # select random batch from objects
        distractor_ints = np.random.choice(len(objects[0]), num_distractor)
        distractors = [(objects[0][i], objects[1][i]) for i in distractor_ints]

        for obj, m in distractors:
            self._position_distractor(obj, m)

    def _position_distractor(self, obj, mask):
        y_max, x_max = self.Background.shape[:2]
        dis_h, dis_w = obj.shape[:2]
        obj_y, obj_x = self._get_random_position(y_max, x_max, dis_h, dis_w)
        local_bg = self.Toy[obj_y:obj_y + dis_h, obj_x:obj_x + dis_w]
        obj = self._adjust_object_brightness(obj, local_bg)
        radius = self._calculate_blur_radius(mask)
        padding = int(2 * radius)
        top_pad, bottom_pad, left_pad, right_pad = self._calculate_padding(obj_y, obj_x, dis_h, dis_w, y_max, x_max,
                                                                           padding)
        fully_padded = self._create_padded_background(obj_y, obj_x, dis_h, dis_w, top_pad, bottom_pad, left_pad,
                                                      right_pad)
        mask_padded, obj_padded = self._pad_object_and_mask(mask, obj, top_pad, bottom_pad, left_pad, right_pad)
        local_bg_new = self._place_object_on_background(fully_padded, mask_padded, obj_padded)
        blurred, mask_blurred = self._apply_blur(local_bg_new, mask_padded, radius)
        self._blend_object_with_background(obj_y, obj_x, top_pad, left_pad, fully_padded, blurred, mask_blurred)

    def _position_objects(self, objects, blending) -> None:
        y_max, x_max = self.Background.shape[:2]
        toy_occupied = np.zeros(self.Background.shape[:2], dtype=bool)
        instance_field = np.zeros(self.Background.shape[:2], dtype=np.int64)
        instance_counter = 1
        for obj_image, obj_mask, class_idx in zip(*objects):
            if self.augmentation:
                obj_image, obj_mask = augmentations(obj_image, obj_mask, self.Background.shape[:2],
                                                    params=self.augmentation)
            assert obj_image.shape[:2] == obj_mask.shape, f'shape of object {class_idx} and masks do not match'
            obj_height, obj_width = obj_image.shape[:2]
            min_obj_height = int(obj_height * 0.7)
            min_obj_width = int(obj_width * 0.7)
            for _ in range(100):
                # get random position
                obj_y, obj_x = self._get_random_position(y_max, x_max, min_obj_height, min_obj_width)

                # adjust object in case it overlaps with border
                final_obj_height, final_obj_width = self._get_final_object_size(obj_y, obj_x, obj_height, obj_width,
                                                                                y_max, x_max)
                mask_temporary = obj_mask[:final_obj_height, :final_obj_width]

                total_area = np.count_nonzero(mask_temporary == 255)

                # check if placement is valid (overlap between objects less then maximum overlap defined
                if not self._is_placement_valid(obj_y, obj_x, final_obj_height, final_obj_width, mask_temporary,
                                                toy_occupied, instance_field, total_area, class_idx):
                    continue

                non_zero_mask = obj_mask[:final_obj_height, :final_obj_width] > 0
                toy_occupied[obj_y:obj_y + final_obj_height, obj_x:obj_x + final_obj_width] |= non_zero_mask

                # blending of object
                if blending:
                    self._blend_object(obj_image, obj_mask, obj_y, obj_x, final_obj_height, final_obj_width)
                else:
                    self._place_object(obj_image, obj_y, obj_x, final_obj_height, final_obj_width)

                self._update_label(obj_mask, class_idx, obj_y, obj_x, final_obj_height, final_obj_width)
                instance_field[obj_y:obj_y + final_obj_height, obj_x:obj_x + final_obj_width][
                    non_zero_mask] = instance_counter
                self.instance_areas[instance_counter] = total_area
                instance_counter += 1
                self._save_instance_mask(obj_mask, obj_y, obj_x, final_obj_height, final_obj_width)
                if len(self.adjusted_objects) > 0:
                    for obj in self.adjusted_objects:
                        instance_mask_new = np.zeros(self.Background.shape[:2], dtype='uint8')
                        instance_mask_new[np.where(instance_field == obj)] = self.class_colors[
                            find_matching_key(self.class_colors, objects[2][obj - 1])]
                        self.InstanceMasks[obj - 1] = instance_mask_new
                break

    @staticmethod
    def create_bg(shape) -> np.ndarray:
            h, w = np.random.randint(800, 1000, 2)
            bg = np.zeros([h, w, 3], dtype='uint8')
            bg += random.randint(0, 10)
            if shape == 3:
                bg = np.dstack([bg]*3)
            return bg

    def read_bg(self, shape) -> np.ndarray:
        if self.bg_folder:
            bg_files = get_image_files(self.bg_folder)
            bg_file = random.choice(bg_files)
            bg = io.imread(bg_file)

            # small background augmentation
            bg[bg > 7] += random.randint(0, 10)
            bg = flip_image(bg, np.random.choice([0, 1, ]))

        else:
            bg = self.create_bg(shape)
        return bg

    @staticmethod
    def _get_final_object_size(obj_y, obj_x, obj_height, obj_width, y_max, x_max):
        final_obj_height = min(y_max - obj_y, obj_height)
        final_obj_width = min(x_max - obj_x, obj_width)
        return final_obj_height, final_obj_width

    def _is_placement_valid(self, obj_y, obj_x, obj_height, obj_width, obj_mask, toy_occupied, instance_field,
                            total_area, class_idx):
        toy_region = toy_occupied[obj_y:obj_y + obj_height, obj_x:obj_x + obj_width]
        overlapping_pixel = np.sum((toy_region + obj_mask / 255) > 1)
        overlap_newobject = overlapping_pixel / total_area
        if overlap_newobject > self.max_overlap:
            return False
        elif self.max_overlap > overlap_newobject > 0.0:
            proposed_area = np.zeros_like(toy_occupied, dtype=np.int64)
            proposed_area[obj_y:obj_y + obj_height, obj_x:obj_x + obj_width] = obj_mask
            overlapping = (instance_field.copy() + proposed_area)
            overlapping_objects = np.unique(overlapping[overlapping > 255])
            overlapping_objects = [o - 255 for o in overlapping_objects]
            for obj in overlapping_objects:
                already_occluded = self.instance_areas[obj] - np.count_nonzero(instance_field == obj)
                overlap_obj = np.count_nonzero(overlapping == (255 + obj))
                overlap_placed_object = (overlap_obj + already_occluded) / self.instance_areas[obj]
                if overlap_placed_object > self.max_overlap:
                    return False
                if self.update_overlapping_masks[class_idx.split('_')[0]]:
                    self.adjusted_objects.append(obj)
            return True
        else:
            return True

    def _blend_object(self, obj_image, obj_mask, obj_y, obj_x, obj_height, obj_width):
        obj_image = subtract_background(obj_image, obj_mask)
        rgb = True if len(obj_image.shape) == 3 else False
        blending_mask = create_blending_mask(obj_mask, rgb)
        blending_mask = blending_mask[:obj_height, :obj_width]
        local_bg = self.Toy[obj_y:obj_y + obj_height, obj_x:obj_x + obj_width]
        bg_s = cv2.multiply(obj_image[:obj_height, :obj_width].astype(np.float64), blending_mask)
        blended_toy = cv2.add(local_bg.astype(np.float64), bg_s)
        blended_toy = np.clip(blended_toy, 0, 255).astype(np.uint8)
        alpha = np.random.uniform(0.5, 1)
        blended_region = self._alpha_blend(blended_toy, local_bg, alpha)
        self.Toy[obj_y:obj_y + obj_height, obj_x:obj_x + obj_width] = blended_region

    def _place_object(self, obj_image, obj_y, obj_x, obj_height, obj_width):
        obj_final = obj_image[:obj_height, :obj_width]
        self.Toy[obj_y:obj_y + obj_height, obj_x:obj_x + obj_width] = np.where(obj_final != 0,
                                                                               self.Toy[obj_y:obj_y + obj_height,
                                                                               obj_x:obj_x + obj_width], obj_final)

    def _update_label(self, obj_mask, class_idx, obj_y, obj_x, obj_height, obj_width):
        obj_mask[obj_mask == 255] = self.class_colors[find_matching_key(self.class_colors, class_idx)]
        mask_indices = obj_mask[:obj_height, :obj_width] > 0
        self.Label[obj_y:obj_y + obj_height, obj_x:obj_x + obj_width][mask_indices] = obj_mask[:obj_height, :obj_width][
            mask_indices]

    def _save_instance_mask(self, obj_mask, obj_y, obj_x, obj_height, obj_width):
        instance_mask = np.zeros(self.Background.shape[:2], dtype='uint8')
        instance_mask[obj_y:obj_y + obj_height, obj_x:obj_x + obj_width] += obj_mask[:obj_height, :obj_width]
        self.InstanceMasks.append(instance_mask)

    @staticmethod
    def _alpha_blend(obj_image, background, alpha=0.5):
        # Ensure alpha is between 0 and 1
        alpha = np.clip(alpha, 0, 1)
        blended = alpha * obj_image + (1 - alpha) * background
        return blended.astype(np.uint8)

    @staticmethod
    def _get_random_position(y_max, x_max, obj_height, obj_width):
        obj_y = np.random.randint(0, max(y_max - obj_height, 1))
        obj_x = np.random.randint(0, max(x_max - obj_width, 1))
        return obj_y, obj_x

    @staticmethod
    def _adjust_object_brightness(obj, local_bg):
        return adjust_object_brightness(obj, local_bg)

    @staticmethod
    def _calculate_blur_radius(mask):
        area = np.sum(mask > 0)
        c2 = np.random.randint(2, 6, 1)
        c1 = 0.2
        radius = int(c1 * np.sqrt(area) + c2 * np.log(area + 1))
        return radius

    @staticmethod
    def _calculate_padding(obj_y, obj_x, obj_height, obj_width, y_max, x_max, padding):
        top_pad = padding if obj_y - padding >= 0 else obj_y
        bottom_pad = padding if y_max >= (obj_y + obj_height + padding) else y_max - obj_y - obj_height
        left_pad = padding if obj_x - padding >= 0 else obj_x
        right_pad = padding if x_max >= (obj_x + obj_width + padding) else x_max - obj_x - obj_width
        return top_pad, bottom_pad, left_pad, right_pad

    def _create_padded_background(self, obj_y, obj_x, obj_height, obj_width, top_pad, bottom_pad, left_pad, right_pad):
        top_padding = self.Toy[(obj_y - top_pad):obj_y, obj_x:obj_x + obj_width]
        bottom_padding = self.Toy[(obj_y + obj_height):obj_y + obj_height + bottom_pad, obj_x:obj_x + obj_width]
        top_bottom_padded = np.vstack(
            [top_padding, self.Toy[obj_y:obj_y + obj_height, obj_x:obj_x + obj_width], bottom_padding])
        left_padding = self.Toy[(obj_y - top_pad):obj_y + bottom_pad + obj_height, (obj_x - left_pad):obj_x]
        right_padding = self.Toy[(obj_y - top_pad):obj_y + bottom_pad + obj_height,
                        obj_x + obj_width:obj_x + obj_width + right_pad]
        fully_padded = np.hstack([left_padding, top_bottom_padded, right_padding])
        return fully_padded

    @staticmethod
    def _pad_object_and_mask(mask, obj, top_pad, bottom_pad, left_pad, right_pad):
        mask_padded = cv2.copyMakeBorder(mask, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
        obj_padded = cv2.copyMakeBorder(obj, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
        return mask_padded, obj_padded

    @staticmethod
    def _place_object_on_background(fully_padded, mask_padded, obj_padded):
        local_bg_new = fully_padded.copy()
        local_bg_new[np.where(mask_padded != 0)] = obj_padded[np.where(mask_padded != 0)]
        return local_bg_new

    @staticmethod
    def _apply_blur(local_bg_new, mask_padded, radius):
        blurred = bokeh_blur(local_bg_new, radius=radius)
        mask_blurred = bokeh_blur(mask_padded, radius=radius)
        return blurred, mask_blurred

    def _blend_object_with_background(self, obj_y, obj_x, top_pad, left_pad, fully_padded, blurred, mask_blurred):
        alpha = mask_blurred / 255.0
        if len(blurred.shape) == 3:
            alpha = np.dstack([alpha] * 3)
        local_bg_new = fully_padded + alpha * blurred

        # Clip the values to 255
        local_bg_new = np.clip(local_bg_new, 0, 255).astype(np.uint8)

        self.Toy[obj_y - top_pad: obj_y - top_pad + local_bg_new.shape[0],
        obj_x - left_pad:obj_x - left_pad + local_bg_new.shape[1]] = local_bg_new


if __name__ == '__main__':
    # create test instance of image creator class
    test = ArtificialImageCreator(class_colors={'circle': 8, 'rectangle': 57, 'triangle': 112}, augmentation={
        'Scale': {'p': 70, 'range': (0.4, 1.8)}, 'RandomRotatex90': {'p': 70}, 'Blur': {'p': 50, 'factor': 5},
        'Flip': {'p': 60}}, update_overlapping_masks={'circle': 8, 'rectangle': 57, 'triangle': 112}, path_bg=None,
                                  max_overlap=0.10, testing_mode=False)
    # create batch of random geometric objects
    batch_size = random.randint(10, 25)
    batch = random_objects(batch_size)

    # image creation and plotting
    test.create_synthetic_image(batch)
    image, label = test.Toy, test.Label
    instance_masked = test.InstanceMasks  #
    plot_instance_inline(image, instance_masked, {'circle': 1, 'rectangle': 2, 'triangle': 3})
