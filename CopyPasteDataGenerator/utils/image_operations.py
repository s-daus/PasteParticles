import numpy as np
from random import randint, uniform
import cv2
from skimage.filters import unsharp_mask
from typing import Dict
from skimage.morphology import dilation, disk, closing
from skimage.feature import canny
from scipy.ndimage import distance_transform_edt
from cv2_rolling_ball import subtract_background_rolling_ball
from skimage.filters import gaussian, rank, threshold_otsu


def get_border_values(image: np.ndarray) -> np.ndarray:
    if len(image.shape) > 2:
        image = np.max(image, axis=2)
    border = []
    border += list(image[0, :-1])
    border += list(image[:-1, -1])
    border += list(image[-1, :0:-1])
    border += list(image[::-1, 0])
    return np.array(border)


def zero_mask(mask):
    max_value = np.max(get_border_values(mask))
    mask_new = mask.astype(np.float64) - max_value
    mask_new[mask_new < 0] = 0.
    return mask_new


def contains_lines(image):
    # Convert the image to grayscale
    img = rank.median(image[:, :, 0], disk(5))
    edges = canny(img, low_threshold=0.08 * np.max(img), high_threshold=0.15 * np.max(img))
    lines = cv2.HoughLinesP(edges.astype(np.uint8), 1, np.pi / 2, 2, None, 30, 1)
    return lines is not None


def adjust_object_brightness(obj: np.ndarray, blending_image: np.ndarray, bright=None, quantile=0.5) -> np.ndarray:
    if bright is None:
        bright = get_border_values(obj)
    obj_adjusted = obj - (np.mean(bright[bright != 0]) - np.quantile(blending_image, quantile))
    obj_adjusted[obj_adjusted < 0] = 0
    obj_adjusted[obj_adjusted > 255] = 255
    obj_adjusted = obj_adjusted.astype(np.uint8)
    return obj_adjusted


def cut_border(obj, mask):
    for i in range(9):
        mask = dilation(mask, disk(5))
    obj[mask == 0] = 0
    return obj


def rotate_image(image: np.ndarray, rotate: int) -> np.ndarray:
    rotated = np.rot90(image, rotate)
    return rotated


def flip_image(image: np.ndarray, direction: int) -> np.ndarray:
    flipped = cv2.flip(image, direction)
    return flipped


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    rescaled = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                          interpolation=cv2.INTER_LINEAR)
    return rescaled


def scale_mask(image: np.ndarray, scale: float) -> np.ndarray:
    rescaled = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                          interpolation=cv2.INTER_NEAREST)
    return rescaled


def augmentations(image: np.ndarray, binary: np.ndarray, bg_shape: tuple, params: Dict[str, any]) -> tuple:
    # define range for operations
    rotate = np.random.choice([-1, 0, 1, 2])

    # actual range of scaling is calculated, for simplification the object size is estimated using the bounding box
    bb = cv2.boundingRect(binary)
    min_size, max_size = min(bb[2:]), max(bb[2:])
    min_size_limit, max_size_limit = 9, int(1 / 3 * min(bg_shape))
    scaling_range = (max(params['Scale']['range'][0], min_size_limit / min_size),
                     min(max_size_limit / max_size, params['Scale']['range'][1]))
    scaling_factor = round(uniform(*scaling_range), 2)

    blur_or_sharpen = randint(-params['Blur']['factor'], params['Blur']['factor'])
    flip = np.random.choice([-1, 0, 1])

    # define probability for operations
    probability_rotation = params['RandomRotatex90']['p']
    probability_scaling = params['Scale']['p']
    probability_blur = params['Blur']['p']
    probability_flip = params['Flip']['p']

    # operation application
    if randint(0, 100) > (100 - probability_rotation):
        image = rotate_image(image, rotate)
        binary = rotate_image(binary, rotate)

    if randint(0, 100) > (100 - probability_flip):
        image = flip_image(image, flip)
        binary = flip_image(binary, flip)

    if randint(0, 100) > (100 - probability_scaling):
        image = scale_image(image, scaling_factor)
        binary = scale_mask(binary, scaling_factor)

    # if value < 0 -> blurring, else sharpening
    if randint(0, 100) > (100 - probability_blur):
        if blur_or_sharpen < 0:
            image_inter = cv2.blur(image, (abs(blur_or_sharpen), abs(blur_or_sharpen)))
            image = image_inter
        else:
            if len(image.shape) == 3:
                image_inter = unsharp_mask(image, radius=abs(blur_or_sharpen), amount=1, channel_axis=-1)
            else:
                image_inter = unsharp_mask(image, radius=abs(blur_or_sharpen), amount=1)
            image_inter = (255 * image_inter)
            image_inter[image_inter < 0] = 0
            image_inter[image_inter > 255] = 255
            image_inter = image_inter.astype(np.uint8)
            image = image_inter
    return image, binary


def generate_mask(img: np.ndarray, dilate=False) -> np.ndarray:
    img = rank.median(img, disk(3))
    thresh = threshold_otsu(img)  # thresholding
    binary = np.zeros(img.shape, np.uint8)
    np.putmask(binary, img > thresh, 255)
    binary = closing(binary, disk(3))
    if dilate:
        binary = dilation(binary, disk(5))
    return binary


def random_circle():
    radius_range = (10, 40)
    rad = np.random.randint(*radius_range)
    image = np.ones([2 * rad + 5, 2 * rad + 5, 3], dtype='uint8')
    mask = np.zeros([2 * rad + 5, 2 * rad + 5], dtype='uint8')
    center = (image.shape[0] // 2, image.shape[1] // 2)
    cv2.circle(image, center=center, radius=rad, color=(255, 0, 0), thickness=-1)
    cv2.circle(mask, center=center, radius=rad, color=255, thickness=-1)
    return image, mask


def random_triangle():
    point_range = (0, 100)
    while True:
        points = np.array([[randint(*point_range) for _ in range(2)] for _ in range(3)], np.int32)
        # checking that area is not too close to zero (points collinear)
        area = abs((points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - (points[2][0] - points[0][0]) * (
                points[1][1] - points[0][1]))
        if area > 5:
            break
    size = np.max(points) + 10
    image = np.ones([size, size, 3], dtype='uint8')
    mask = np.zeros([size, size], dtype='uint8')
    cv2.fillPoly(image, [points], (0, 255, 0))
    cv2.fillPoly(mask, [points], 255)
    return image, mask


def random_rectangle():
    size_range = (20, 100)
    w, h = [randint(*size_range) for _ in range(2)]
    start_point = (8, 8)
    size = max([w, h]) + 10
    image = np.ones([size, size, 3], dtype='uint8')
    mask = np.zeros([size, size], dtype='uint8')
    cv2.rectangle(image, start_point, (start_point[0] + h, start_point[1] + w), (0, 0, 255), -1)
    cv2.rectangle(mask, start_point, (start_point[0] + h, start_point[1] + w), 255, -1)
    return image, mask


def random_objects(number_of_objects: int) -> tuple:
    object_dict = {1: 'triangle', 2: 'circle', 3: 'rectangle'}
    objects = []
    images = []
    masks = []
    for i in range(number_of_objects):
        n = np.random.choice([1, 2, 3])
        shape = object_dict[n]
        objects.append(shape)
        if n == 1:
            image, mask = random_triangle()

        elif n == 2:
            image, mask = random_circle()
        else:
            image, mask = random_rectangle()
        masks.append(mask)
        images.append(image)
    return images, masks, objects


def create_blending_mask(mask, rgb, dilate=False):
    if dilate:
        mask = dilation(mask, disk(5))

    binary = (mask / 255).astype(np.uint8)
    dist_transform = distance_transform_edt(1 - binary)

    boundary = np.logical_xor(binary, np.roll(binary, shift=1, axis=0)) | np.logical_xor(binary,
                                                                                         np.roll(binary, shift=1,
                                                                                                 axis=1))
    y_coords, x_coords = np.nonzero(boundary)

    min_to_top = y_coords.min()
    min_to_bottom = binary.shape[0] - 1 - y_coords.max()
    min_to_left = x_coords.min()
    min_to_right = binary.shape[1] - 1 - x_coords.max()
    shortest_distance = min(min_to_top, min_to_bottom, min_to_left, min_to_right)

    dist_transform = dist_transform / (shortest_distance + 1)
    alpha = 1.0 / (dist_transform + 0.5) ** 1.2  # factor 0.5 to avoid divsion by zero
    blurred_mask = gaussian(alpha, sigma=8)
    blurred_mask = blurred_mask / blurred_mask.max()
    if rgb:
        blurred_mask = np.dstack([blurred_mask]*3)
    return blurred_mask


# def subtract_background(image, mask=None):
#     if mask is not None:
#         area = np.count_nonzero(mask)
#         radius = np.sqrt(area / np.pi) + 50
#     else:
#         radius = np.max(image.shape) - 1 / 10 * np.max(image.shape)
#     img, background = subtract_background_rolling_ball(image, int(radius), light_background=False,
#                                                        use_paraboloid=False, do_presmooth=True)
#     img_smoothed = rank.median(img, disk(3))
#     return img_smoothed


def subtract_background(image, mask=None):
    if mask is not None:
        area = np.count_nonzero(mask)
        radius = np.sqrt(area / np.pi) + 50
    else:
        radius = np.max(image.shape) - 1 / 10 * np.max(image.shape)
    if len(image.shape) == 2:
        # The image is grayscale
        img, background = subtract_background_rolling_ball(image, int(radius), light_background=False,
                                                           use_paraboloid=False, do_presmooth=True)
    else:
        # The image is RGB, process each channel
        blue, green, red = cv2.split(image)
        foreground_blue, bg_blue = subtract_background_rolling_ball(blue, int(radius), light_background=False,
                                                                    use_paraboloid=False, do_presmooth=True)
        foreground_green, bg_green = subtract_background_rolling_ball(green, int(radius), light_background=False,
                                                                      use_paraboloid=False, do_presmooth=True)
        foreground_red, bg_red = subtract_background_rolling_ball(red, int(radius), light_background=False,
                                                                  use_paraboloid=False, do_presmooth=True)
        # Merge the channels back
        img = cv2.merge((foreground_blue, foreground_green, foreground_red))
    img_smoothed = cv2.GaussianBlur(img, (3, 3), 0)
    return img_smoothed


def disk_kernel(radius):
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2
    return mask.astype(np.float32)


def bokeh_blur(image, radius=7):
    # Create disk-shaped kernel
    kernel = disk_kernel(radius)
    kernel = kernel / kernel.sum()

    # Blur the entire image
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred
