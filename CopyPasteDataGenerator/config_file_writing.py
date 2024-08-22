from collections import OrderedDict
import json

conf = dict(
    # data
    classes_wo_bg=2,  # number of classes excluding background class
    class_names=('particle', 'bubble'),
    path_bg="image_database/background",
    bg_extension='*.tif',
    path_toys="image_database/objects",  # images per class should either be stored in different sub folder
    # or be distinguishable by name if in same folder
    path_masks="image_database/masks",  # same pattern as for toy masks if binary data available, if not,
    image_extension='*.tif',

    # image generation
    type='instance',  # define which type of training data (instance or semantic)
    number_images=5,
    target_imagesize=(2056, 2464),

    # set either objects_numbers or area_fraction
    objects_numbers=None,  # number of objects to be used for image generation as dict of class name:
    # (lower range of objects, upper range of objects) for example {'particle': (5, 50), 'bubble': (1,  3)}

    area_fraction=0.1,  # area fraction to be filled with objects
    object_area_dataframe_path='mask_areas.csv',  # for area fraction csv file necessary that contains path of mask

    # whether occluded part of the overlapping masks should be cut from the mask: True means that occluded regions
    # will be cut from the ground truth
    update_overlapping_masks=OrderedDict({'particle': True, 'bubble': False}),

    # number or range of objects
    object_split=OrderedDict({'particle': 0.9, 'bubble': 0.1}),
    class_colors=OrderedDict({'particle': 1, 'bubble': 2}),
    max_overlap=0.3,
    augmentation=OrderedDict({
        'Scale': {'p': 0, 'range': (0.5, 2)},  # to make the approach more general the limit for the actual objects
        # were set according to ISO 13322-2 (min size: 9 pixels in one direction, max: one third of shorter side of
        # the image)
        'Flip': {'p': 60},
        'RandomRotatex90': {'p': 70},  # random rotation my multiples of 90 degrees (90, 180, 270)
        'Blur': {'p': 0, 'factor': 2}  # gaussian blur
    }),
    plotting_results=True,

    # saving
    save_path=r"synthetic_images",
    save_extension='.tif')

with open("config_file.json", "w") as outfile:
    json.dump(conf, outfile)
