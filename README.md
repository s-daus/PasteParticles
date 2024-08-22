[![Paper (Powder Technology)](https://doi.org/)]

# Synthetic Data Generation with Copy-Paste Augmentation

The subproject contains the python project for synthetic data generation using Copy-Paste augmentation. The image data properties like area fraction to be filled with objects can be adjusted to create synthetic data for specific purposes. Since the ground truth 
 
## Setup

Environment can be created using the .toml file that is included. 
Simply run 
```
poetry install
```

## Configuration

The generation process is defined by parameters in the config_file.json. You can adjust these parameters directly within the JSON file or use config_file_writing.py to programmatically generate a new configuration. The script also contains further information on the purpose of each parameter. The config_file.json contains default values for all parameters. 
Key Parameters:

    Image Database: The image_database folder contains instances of bubbles and particles as used in our paper. These instances form the basis for synthetic image generation.
    Mask Areas: For new image databases, it's necessary to create a mask_areas.csv file using create_maskareas_csv.py. This file records the pixel area of each mask, heping to efficiently select  objects based on the desired area fraction of objects in the generated images.


## Advanced Usage

    Custom Datasets: To use a custom dataset for image generation, ensure that your object images and masks are properly formatted and placed in a similar structure as the provided image_database. Update the config_file.json to point to your custom dataset directory.
    Parameter Tuning: Experiment with different settings in config_file.json to achieve desired outcomes in synthetic image properties, such as object density, augmentation effects, and image size.

## Testing and Examples

    Included Data: Utilize the included bubble and particle instances to replicate the data generation process described in the paper.
    Custom Experiments: Consider creating a small, controlled dataset to experiment with various configuration settings and understand their impact on the generated synthetic images.
		