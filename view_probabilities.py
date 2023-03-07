# Created using chatgpt

import numpy as np
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import simulationFuncs as SF
from tqdm import tqdm

# Directory where the .npy files are located
data_dir = "/nfs/a1/homes/py15jmc/bootstrap/2023/timeseries/"

# Get a list of all .npy files in the directory that start with "probabilities"
file_list = [f for f in os.listdir(data_dir) if f.endswith(".npy") and f.startswith("probabilities")]

# Sort the list by index 1
file_list = sorted(file_list, key=lambda x: int(x.split("_")[1]))

# Create an empty array to hold the data
data = np.empty((len(file_list),), dtype=np.ndarray)

# Loop through each file and load the data
for i, filename in enumerate(file_list):
    filepath = os.path.join(data_dir, filename)
    data[i] = np.load(filepath)

# Stack the arrays along axis 0 to create a 3D array
result = np.stack(data, axis=0)
np.save(data_dir + "all.npy", result)
print ("Saved all \n\n\n")

window_size = 10
moving_mean = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='valid'), axis=0, arr=result)

plt.pcolormesh(np.sum((moving_mean > 0.5), axis=0), vmin=0, vmax=moving_mean.shape[0])
plt.colorbar()

wdir = "/nfs/a1/homes/py15jmc/"

SHP_i = np.load(wdir + "bootstrap/shp_i_20150511_20170617.npy")

SHP_j = np.load(wdir + "bootstrap/shp_j_20150511_20170617.npy")

filt_moving_mean = SF.filterPredictions(np.sum((moving_mean > 0.5), axis=0) > 0.5*moving_mean.shape[0], SHP_i, SHP_j, threshold=0.7)

plt.figure()
plt.pcolormesh(filt_moving_mean[1])
plt.show()

# For creating a gif
# Create a list to hold the PNG images
images = []

# Loop through each slice of the array
for i in tqdm(range(result.shape[0])):
    # Create a pcolormesh plot of the slice
    plt.pcolormesh(result[i,:,:]>0.5, vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Slice {}".format(i+80))
    # plt.xlabel("Index 2")
    # plt.ylabel("Index 1")
    
    # Save the plot as a PNG image
    filename = "Figures/slice{}_w10_0.5.png".format(i+80)
    plt.savefig(filename)
    
    # Add the image to the list
    images.append(imageio.imread(filename))
    
    # Clear the plot for the next iteration
    plt.clf()

# Use imageio to create a GIF from the PNG images
imageio.mimsave("Figures/result_w10_0.5.gif", images, duration=1.0)