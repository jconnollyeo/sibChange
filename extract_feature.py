import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# Define the path to the directory containing the CSV and npy files
dir = "/nfs/see-fs-01_users/py15jmc/py15jmc/bootstrap/2023/"
dir = "timeseries/"

# Define the name of the column to be imported from the CSV files
column_name = "jk_bias"

# Get a list of all CSV files in the directory, sorted by ix1
csv_files = sorted(glob.glob(dir + "IFG_*_*_*_*.csv"), key=lambda x: int(x.split("_")[1]))

# Initialize empty lists to store the data
jk_std_list = []
dropped_mask_list = []

# Loop through each CSV file
for csv_file in tqdm(csv_files):
    # Extract the ix1 value from the file name
    ix1 = csv_file.split("_")[1]
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Extract the desired column and append it to the jk_std list
    jk_std_list.append(df[column_name].values)
    
    # Read in the corresponding dropped_mask .npy file
    npy_file = dir + "dropped_mask_" + ix1 + "*.npy"
    npy_files = glob.glob(npy_file)
    
    # Make sure only one .npy file is found
    assert len(npy_files) == 1
    
    # Load the .npy file into a numpy array and append it to the dropped_mask list
    dropped_mask = np.load(npy_files[0])
    dropped_mask_list.append(dropped_mask)

out = np.empty(shape=(len(dropped_mask_list), *dropped_mask_list[0].shape))

for i, (jk, drop) in enumerate(zip(jk_std_list, dropped_mask_list)):
    out[i][drop] = jk

np.save(dir + column_name + "_all.npy", out)