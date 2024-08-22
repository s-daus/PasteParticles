import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.ndimage as ndi
from typing import Dict
from skimage.color import label2rgb
from utils.data_operations import get_key_from_value, get_image_files
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from skimage import io


def plot_semantic(toy_path, label_path, class_colors: dict) -> None:
    toy = io.imread(toy_path)
    label = io.imread(label_path)
    fig, ax = plt.subplots(ncols=2, layout="constrained")
    cmap = get_cmap('viridis')
    patches = []
    norm = Normalize(vmin=0, vmax=max(class_colors.values()))
    for cl, color in class_colors.items():
        patches.append(mpatches.Patch(color=cmap(norm(color)), label=cl))
    ax[1].legend(handles=patches, loc='upper left', prop={'size': 8},
                 facecolor='white', framealpha=0.3)
    ax[0].imshow(toy, cmap='gray')
    ax[1].matshow(label, cmap='viridis')
    for a in ax:
        a.axis('off')
    plt.show()


def plot_instance(toy_path, target_path, class_colors: Dict[str, any]) -> None:
    toy = io.imread(toy_path)
    plot_image = toy.copy()
    targets = np.load(f"{target_path}\\{target_path.stem}" + ".npy")
    masks_names = [img_n for img_n in (get_image_files(target_path))]
    masks = [io.imread(img_n) for img_n in masks_names]
    masks_stacked = np.dstack(masks)
    labeled_coins, _ = ndi.label(masks_stacked)
    if len(labeled_coins.shape) == 3:
        labeled_coins = np.max(labeled_coins, axis=-1)
    image_label_overlay = label2rgb(labeled_coins, image=plot_image, bg_label=0)
    for img, name in zip(masks, masks_names):
        box = cv2.boundingRect(img)
        cl = get_key_from_value(class_colors, targets[int(name.stem)])
        cv2.rectangle(image_label_overlay, box[:2], (box[0] + box[2], box[1] + box[3]), (1., 1., 1.), thickness=1)
        cv2.putText(image_label_overlay, cl, (box[0], box[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (1., 1., 1.))
    fig, ax = plt.subplots(ncols=1, layout="constrained")
    ax.imshow(image_label_overlay)
    ax.axis('off')
    plt.show(block=True)


def plot_instance_inline(toy, masks, class_colors: Dict[str, any]) -> None:
    plot_image = toy.copy()
    masks_stacked = np.dstack(masks)
    labeled_coins, _ = ndi.label(masks_stacked)
    if len(labeled_coins.shape) == 3:
        labeled_coins = np.max(labeled_coins, axis=-1)
    image_label_overlay = label2rgb(labeled_coins, image=plot_image, bg_label=0)
    for img in masks:
        box = cv2.boundingRect(img)
        cl = get_key_from_value(class_colors, np.unique(img[img!=0]))
        cv2.rectangle(image_label_overlay, box[:2], (box[0] + box[2], box[1] + box[3]), (1., 1., 1.), thickness=1)
        cv2.putText(image_label_overlay, cl, (box[0], box[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (1., 1., 1.))
    fig, ax = plt.subplots(ncols=1, layout="constrained")
    ax.imshow(image_label_overlay)
    ax.axis('off')
    plt.show(block=True)
