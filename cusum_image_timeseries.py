import numpy as np
# from BMC.functions import detect_cusum
# from detect_cusum_ import detect_cusum
from detecta import detect_cusum
import matplotlib.pyplot as plt

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

def cusum_man(imgs, h=0.5, k=1):
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
            cusum_val[j] = 0

        cusum_arr[:, :, 0] = cusum_val
        cusum_arr[:, :, 1] = change_point

    return cusum_arr

probs = np.load("/nfs/see-fs-01_users/py15jmc/py15jmc/bootstrap/2023/timeseries/all.npy")
jk_std = np.load("/nfs/see-fs-01_users/py15jmc/py15jmc/bootstrap/2023/timeseries/jk_std_all.npy")
jk_bias = np.load("/nfs/see-fs-01_users/py15jmc/py15jmc/bootstrap/2023/timeseries/jk_bias_all.npy")
# Example usage
# imgs = np.random.rand(10, 100, 100)
# cusum_arr = cusum(probs)
fig, ax = plt.subplots(nrows=1, ncols=2)

cusum_man_arr = cusum_man(probs)
p0 = ax[0].pcolormesh(cusum_man_arr[:, :, 0])
p1 = ax[1].pcolormesh(cusum_man_arr[:, :, 1])
plt.colorbar(p0, ax=ax[0])
plt.colorbar(p1, ax=ax[1])

fig, ax = plt.subplots(nrows=1, ncols=2)
cusum_jk_std = cusum_man(jk_std)
p0 = ax[0].pcolormesh(cusum_jk_std[:, :, 0])
p1 = ax[1].pcolormesh(cusum_jk_std[:, :, 1])
plt.colorbar(p0, ax=ax[0])
plt.colorbar(p1, ax=ax[1])

fig, ax = plt.subplots(nrows=1, ncols=2)
cusum_jk_bias = cusum_man(jk_bias)
p0 = ax[0].pcolormesh(cusum_jk_bias[:, :, 0])
p1 = ax[1].pcolormesh(cusum_jk_bias[:, :, 1])
plt.colorbar(p0, ax=ax[0])
plt.colorbar(p1, ax=ax[1])

plt.show()