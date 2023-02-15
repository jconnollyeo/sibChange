import numpy as np
import matplotlib.pyplot as plt
import sys
# import h5py as h5 

# print (h5.File()["Phase"][:])
# shape = ()

wdir = "/nfs/a1/homes/py15jmc/"
ifgs = np.load(wdir + "complex.npy")

ix1, ix2 = 100, 101

ifg1 = ifgs[ix1]
ifg2 = ifgs[ix2]

shape = ifg1.shape

SHP_i = np.load(wdir + "bootstrap/shp_i_20150511_20170617.npy")
SHP_j = np.load(wdir + "bootstrap/shp_j_20150511_20170617.npy")

mask = ~np.isnan(SHP_i) & ~np.isnan(SHP_j)

# Make some synthetic predictions

predictions = np.random.randint(low=0, high=2, size=np.product(shape)) == 1
for _ in range(5):
    predictions = (predictions*np.random.randint(low=0, high=2, size=np.product(shape)))

predictions = predictions.reshape(shape)


def filterPredictions(predictions, SHP_i, SHP_j, threshold):

    mask_siblings = np.empty_like(SHP_i)

    mask_siblings[mask] = predictions[(SHP_i[mask]).astype(int), (SHP_j[mask]).astype(int)]

    return (np.nansum(mask_siblings, axis=0) > threshold) & predictions

new_predictions = filterPredictions(predictions, SHP_i, SHP_j, 4)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)

ax[0].matshow(predictions)
ax[0].set_title("Predictions")

ax[1].matshow(new_predictions)
ax[1].set_title("Number of sibs with True prediction > 4")

# plt.figure()
# plt.hist(np.nansum(mask_siblings, axis=0).flatten(), bins=np.linspace(-14.5, 100.5, 85))
plt.show()