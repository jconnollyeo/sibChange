import simulationFuncs as SF
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from tqdm import tqdm
import pandas as pd
from scipy import signal
from datetime import datetime
import random
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pathlib import Path
import joblib 
from datetime import datetime

# Read in the data
# Parse the data and create the dataframe used by the RF model
# Read in the model
# Apply the model to the dateframe
# Plot the results (binary) over the top of a amplitude image.

wdir = "/nfs/a1/homes/py15jmc/"
ifgs = np.load(wdir + "complex.npy")

ix1, ix2 = 100, 101

ifg1 = ifgs[ix1]
ifg2 = ifgs[ix2]

SHP_i = np.load(wdir + "bootstrap/shp_i_20150511_20170617.npy")
SHP_j = np.load(wdir + "bootstrap/shp_j_20150511_20170617.npy")

n_siblings = np.sum(~np.isnan(SHP_i) * ~np.isnan(SHP_j), axis=0)

coherence_2015 = np.load(wdir + "bootstrap/coherence_2015.npy")

metrics = SF.generateMetricsIFG(ifg1, ifg2, SHP_i, SHP_j)

df = pd.DataFrame(metrics, columns = np.array(["i", "j", "n_siblings", "jk_std", "jk_bias", "amp_mean", "amp_std", "amp_px", "poi_diff", "max_amp_diff", "actual_coherence", "apparent_coherence"]))
# df = pd.read_csv("/nfs/a1/homes/py15jmc/bootstrap/2023/IFG_100_101_20230214_162302.csv")

dropped_mask = np.ones(ifg1.shape, dtype=bool)

# Remvoe any nan values
for ix in list(set(np.where(np.isnan(df))[0])):
    
    dropped_mask[int(df["i"][ix]), int(df["j"][ix])] = False
    df = df.drop(ix)

# Remove any pixels that have fewer than 15 siblings. 
for ix in list(set(np.where(df["n_siblings"] < 15)[0])):
    
    dropped_mask[int(df["i"][ix]), int(df["j"][ix])] = False
    df = df.drop(ix)

df.to_csv(f"IFG_{ix1}_{ix2}_{datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')}.csv")
RF = joblib.load("RF_20230215_121116.jbl")

predictions = RF.predict(df[["jk_std", "jk_bias", "amp_mean", "max_amp_diff", "poi_diff", "apparent_coherence"]])

predictions_arr = np.zeros(ifg1.shape, dtype=bool)

predictions_arr[dropped_mask] = predictions

np.save("predictions.npy", predictions_arr)
# pred_arr = predictions.reshape(ifg1.shape)

fig, ax = plt.subplots()
ax.matshow(np.mean(abs(ifgs), axis=0), cmap="binary_r")
ax.matshow(SF.falseNaN(predictions_arr), cmap="RdBu")

fig, ax = plt.subplots()
mask_siblings, predictions_filt = SF.filterPredictions(predictions_arr, SHP_i, SHP_j, 5)
ax.matshow(np.mean(abs(ifgs), axis=0), cmap="binary_r")
ax.matshow(SF.falseNaN(predictions_filt), cmap="RdBu")

plt.show()