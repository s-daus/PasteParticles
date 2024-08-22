import json
from utils.data_generation import generate_data, plot_images


with open("config_file.json", "r") as outfile:
    data = json.load(outfile)

if __name__ == '__main__':
    indexes = generate_data(data, data['type'], number_of_cores=7)
    if data["plotting_results"]:
        plot_images(data, data['type'], indexes)
