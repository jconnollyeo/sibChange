import numpy as np
# from BMC.functions import detect_cusum
# from detect_cusum_ import detect_cusum
from detecta import detect_cusum
import matplotlib.pyplot as plt
from tqdm import tqdm 
import simulationFuncs as SF

wdir = "/nfs/a1/homes/py15jmc/"
ifgs = np.load(wdir + "complex.npy")

SHP_i = np.load(wdir + "bootstrap/shp_i_20150511_20170617.npy")
SHP_j = np.load(wdir + "bootstrap/shp_j_20150511_20170617.npy")

def cusum(imgs, h=0.5, k=1):
    """
    Perform CUSUM change detection on a time series of images along axis 0 for each pixel using BMC's detect_cusum function.
    Args:
        imgs: a 3D numpy array with shape (num_imgs, height, width), where num_imgs is the number of images in the time series.
        h: the threshold for detecting changes (default 0.5).
        k: the gain (or sensitivity) parameter (default 1).
    Returns:
        A 3D numpy array with shape (height, width, 2), where the last dimension contains the CUSUM values and the change points for each pixel.
    """
    num_imgs, height, width = imgs.shape
    cusum_arr = np.zeros((height, width, 2))
    cusum_val = np.zeros((height, width))
    change_point = np.zeros((height, width))

    for i in range(1, num_imgs):
        diff = np.abs(imgs[i] - imgs[i-1])
        cusum_val += detect_cusum(diff, h, k)
        cusum_arr[:, :, 0] = cusum_val
        cusum_arr[:, :, 1] = change_point

        # Check for change points
        change_mask = cusum_val > 0
        change_point[change_mask] = i
        cusum_val[change_mask] = 0

    return cusum_arr

def cusum_man(imgs, h=0.5, k=1, max_change_points=None):
    """
    Perform CUSUM change detection on a time series of images along axis 0 for each pixel using BMC's detect_cusum function.
    Args:
        imgs: a 3D numpy array with shape (num_imgs, height, width), where num_imgs is the number of images in the time series.
        h: the threshold for detecting changes (default 0.5).
        k: the gain (or sensitivity) parameter (default 1).
    Returns:
        A 3D numpy array with shape (height, width, 2), where the last dimension contains the CUSUM values and the change points for each pixel.
    """
    num_imgs, height, width = imgs.shape
    cusum_arr = np.zeros((height, width, 2), dtype=object)
    cusum_val = np.zeros((height, width))
    change_point = np.zeros((height, width), dtype=object)

    for i in range(1, num_imgs):
        diff = np.abs(imgs[i] - imgs[i-1])
        # cusum_val += detect_cusum(diff, h, k)
        cusum_val += k * (diff - h - cusum_val)

        # Check for change points
        change_mask = cusum_val > 0
        for j in np.argwhere(change_mask):
            j = tuple(j)
            if change_point[j]:
                # Already found a change point, append the new one
                change_point[j].append(i)
            else:
                # First change point, start a new list
                change_point[j] = [i]

        # Pad lists with NaN values to ensure equal length
        max_len = max(len(p) for p in change_point.flat) if max_change_points is None else max_change_points
        for j, p in np.ndenumerate(change_point):
            if p:
                change_point[j] = np.pad(p, (0, max_len - len(p)), mode='constant', constant_values=np.nan)

        cusum_arr[:, :, 0] = cusum_val
        cusum_arr[:, :, 1] = change_point

    return cusum_arr

def cusum_single(imgs, h=0.5, k=1):
    """
    Perform CUSUM change detection on a time series of images along axis 0 for each pixel using BMC's detect_cusum function.
    Args:
        imgs: a 3D numpy array with shape (num_imgs, height, width), where num_imgs is the number of images in the time series.
        h: the threshold for detecting changes (default 0.5).
        k: the gain (or sensitivity) parameter (default 1).
    Returns:
        A 3D numpy array with shape (height, width, 2), where the last dimension contains the CUSUM values and the change points for each pixel.
    """
    num_imgs, height, width = imgs.shape
    cusum_arr = np.zeros((height, width, 2))
    cusum_val = np.zeros((height, width))
    change_point = np.zeros((num_imgs, height, width), dtype=bool)

    for i in tqdm(range(1, num_imgs)):
        diff = np.abs(imgs[i] - imgs[i-1])
        # cusum_val += detect_cusum(diff, h, k)
        cusum_val += k * (diff - h - cusum_val)
        # Check for change points
        change_mask = cusum_val > 0
        change_point[i][change_mask] = True
        cusum_val[change_mask] = 0

        cusum_arr[:, :, 0] = cusum_val
        # cusum_arr[:, :, 1] = change_point

    return cusum_arr, change_point

def cusum_change_detection(probs, threshold=0.5, drift=0, sub_drift=0, method='mean'):
    """
    Perform CUSUM change detection on a time series of images for each pixel.

    Args:
        probs (numpy.ndarray): A 3D numpy array with shape (time, height, width) containing values between 0 and 1.
        threshold (float): The detection threshold. Default is 0.5.
        drift (float): The drift value. Default is 0.
        sub_drift (float): The sub-drift value. Default is 0.
        method (str): The method used to calculate the mean and standard deviation. Can be 'mean' or 'median'. Default is 'mean'.

    Returns:
        numpy.ndarray: A 3D numpy array with shape (height, width, n_cp_max) containing the CUSUM values and change point indices for each pixel.
    """
    # Determine mean and standard deviation for each pixel
    if method == 'mean':
        mu = np.mean(probs, axis=0, keepdims=True)
        std = np.std(probs, axis=0, keepdims=True)
    elif method == 'median':
        mu = np.median(probs, axis=0, keepdims=True)
        std = np.median(np.abs(probs - mu), axis=0, keepdims=True) * 1.4826
    else:
        raise ValueError("Invalid method. Must be 'mean' or 'median'.")

    # Initialize the CUSUM array with nan
    time, height, width = probs.shape
    n_cp_max = height * width
    cusum_arr = np.empty((time, height, width))
    cusum_arr[:] = np.nan

    # Calculate the CUSUM values for each pixel
    Z = (probs - mu) / std
    cusum = np.maximum(0, np.cumsum(Z, axis=0) - drift)
    cusum -= np.arange(len(probs))[:, None, None] * sub_drift

    # Find change points for each pixel
    change_point = np.zeros((time, height, width))
    change_point[:] = np.nan

    for i in range(time):
        mask = cusum[i, :, :] > threshold
        if np.any(mask):
            indices = np.where(mask)
            start = indices[0][0]
            end = indices[0][-1]
            change_point[i, indices[0], indices[1]] = indices[0]
            change_point[i, start:end+1, indices[1]] = np.nan

    print (cusum.shape)

    # cusum_arr[:, :, :time] = np.swapaxes(cusum, 1, 2).T
    cusum_arr[:, :, :] = cusum

    change_point_arr = np.swapaxes(change_point, 1, 2).T
    change_point_arr[np.isnan(change_point_arr)] = 0
    return np.concatenate([cusum_arr, change_point_arr], axis=2)

probs = np.load("/nfs/see-fs-01_users/py15jmc/py15jmc/bootstrap/2023/timeseries/all.npy")
jk_std = np.load("/nfs/see-fs-01_users/py15jmc/py15jmc/bootstrap/2023/timeseries/jk_std_all.npy")
jk_bias = np.load("/nfs/see-fs-01_users/py15jmc/py15jmc/bootstrap/2023/timeseries/jk_bias_all.npy")

# Example usage
# imgs = np.random.rand(10, 100, 100)
# cusum_arr = cusum(probs)
# fig, ax = plt.subplots(nrows=1, ncols=2)

# cusum_man_arr = cusum_man(probs)
# cusum_jk_std = cusum_man(jk_std)
# cusum_jk_bias = cusum_man(jk_bias)

# cusum_man_arr = cusum_change_detection(probs)
# cusum_jk_std = cusum_change_detection(jk_std)
# cusum_jk_bias = cusum_change_detection(jk_bias)

cusum_man_arr = cusum_single(probs)
cusum_jk_std = cusum_single(jk_std)
cusum_jk_bias = cusum_single(jk_bias)

# SF.animate_slider(cusum_jk_bias[1])

filt_cusum = np.empty(cusum_jk_bias[1].shape)
for i in tqdm(np.arange(cusum_jk_bias[1].shape[0])):
    filt_cusum[i] = SF.filterPredictions(cusum_jk_bias[1][i], SHP_i, SHP_j, threshold=0.1)[1]

# p0 = ax[0].pcolormesh(cusum_man_arr[:, :, 0])
# p1 = ax[1].pcolormesh(cusum_man_arr[:, :, 1])
# plt.colorbar(p0, ax=ax[0])
# plt.colorbar(p1, ax=ax[1])

# fig, ax = plt.subplots(nrows=1, ncols=2)
# p0 = ax[0].pcolormesh(cusum_jk_std[:, :, 0])
# p1 = ax[1].pcolormesh(cusum_jk_std[:, :, 1])
# plt.colorbar(p0, ax=ax[0])
# plt.colorbar(p1, ax=ax[1])

# fig, ax = plt.subplots(nrows=1, ncols=2)
# p0 = ax[0].pcolormesh(cusum_jk_bias[:, :, 0])
# p1 = ax[1].pcolormesh(cusum_jk_bias[:, :, 1])
# plt.colorbar(p0, ax=ax[0])
# plt.colorbar(p1, ax=ax[1])

# plt.show()