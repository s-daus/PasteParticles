[Paper (Powder Technology)] (https://doi.org/10.1016/j.powtec.2024.119884)

# Synthetic Data Generation with Copy-Paste Augmentation

This repository belongs to the research paper "In-line image analysis of particulate processes with deep learning: Optimizing training data generation via copy-paste augmentation." It provides the Python scripts to generate synthetic image data for training semantic or instance segmentation models for in-line particle data. The idea is to build an image database by extracting object instances at low solid concentrations where particles are easier to detect. The image database is used to generate synthetic images at higher concentrations with ground truth labels via Copy-Paste augmentation. The resulting data can be used directly to train semantic and instance segmentation models. This repository only contains the scripts for the synthetic image generation based on an existing image database. For more information regarding obtaining the image database with conventional image segmentation methods and training a Mask R-CNN model, see the additional supplementary data of the publication: [here](https://opara.zih.tu-dresden.de/items/26ec6a28-37ef-4066-bead-2e955cbf1c34)
 
## Setup

Environment can be created using the .toml file that is included. 
Simply run 
```
poetry install
```

## Configuration

The generation process is defined by parameters in the config_file.json. You can adjust these parameters directly within the JSON file or use config_file_writing.py to generate a new configuration. The script also contains further information on the purpose of each parameter. The config_file.json contains default values for all parameters. 
Key Parameters:

Image Database: The image_database folder contains instances of bubbles and particles as used in our paper. These instances form the basis for synthetic image generation.

Mask Areas: For new image databases, it's necessary to create a mask_areas.csv file using create_maskareas_csv.py. This file records the pixel area of each mask, heping to efficiently select objects based on the desired area fraction of objects in the generated images.


## Advanced Usage

Custom Datasets: To use a custom dataset for image generation, ensure that your object images and masks are properly formatted and placed in a similar structure as the provided image_database. Update the config_file.json to point to your custom dataset directory.

Parameter Tuning: Experiment with different settings in config_file.json to achieve desired outcomes in synthetic image properties, such as object density, augmentation effects, and image size.

## Testing and Examples

Included Data: Utilize the included bubble and particle instances to replicate the data generation process described in the paper.

Custom Experiments: Consider creating a small, controlled dataset to experiment with various configuration settings and understand their impact on the generated synthetic images.
		