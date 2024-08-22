from pathlib import Path
from skimage import io
import pandas as pd

mask_folder = Path(r"image_database/masks")

csv_file = pd.DataFrame(columns=['masks', 'areas'])

mask_paths = []
mask_areas = []

for file in mask_folder.glob('*.tif'):
    mask_paths.append(Path(mask_folder, file.name))
    mask = io.imread(file)
    mask_areas.append(len(mask[mask == 255]))

csv_file['masks'] = mask_paths
csv_file['areas'] = mask_areas

csv_file.to_csv("mask_areas.csv")