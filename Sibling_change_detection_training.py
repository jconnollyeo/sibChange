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


def importData():
    wdir = "/nfs/a1/homes/py15jmc/"
    ifgs = np.load(wdir + "complex.npy")

    dates = np.load(wdir + "dates.npy")

    SHP_i = np.load(wdir + "bootstrap/shp_i_20150511_20170617.npy")

    SHP_j = np.load(wdir + "bootstrap/shp_j_20150511_20170617.npy")

    n_siblings = np.sum(~np.isnan(SHP_i) * ~np.isnan(SHP_j), axis=0)

    coherence_2015 = np.load(wdir + "bootstrap/coherence_2015.npy")

    return wdir, ifgs, dates, SHP_i, SHP_j, n_siblings, coherence_2015


def sampleUniform2D(arr1, arr2, mask=None, bins=30, N=1000):
    import random
    import numpy as np
    from tqdm import tqdm

    if mask is not None:
        mask_flatten = mask.flatten()
    else:
        pass

    assert arr1.shape == arr2.shape, "arr1 and arr2 must be the same shape."

    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()

    min1, max1 = np.nanmin(arr1), np.nanmax(arr1)
    min2, max2 = np.nanmin(arr2), np.nanmax(arr2)

    bins1 = np.linspace(min1, max1, bins)
    bins2 = np.linspace(min2, max2, bins)

    bin_numbers1 = np.digitize(arr1_flat, bins=bins1, right=True)
    bin_numbers2 = np.digitize(arr2_flat, bins=bins2, right=True)

    N_each_bin = int(np.ceil(N / ((bins1.size - 2) * (bins2.size - 2))))

    out = np.empty((np.max(bin_numbers1 + 1) * np.max(bin_numbers2 + 1), N_each_bin))

    p = np.arange(arr1.size, dtype=float)

    i = 0

    for b1 in tqdm(np.arange(np.max(bin_numbers1) + 1)):
        for b2 in np.arange(np.max(bin_numbers2) + 1):
            mask_bin = (bin_numbers1 == b1) & (bin_numbers2 == b2) & (mask_flatten)
            if np.sum(mask_bin) > N_each_bin:
                picked_p = random.sample(list(p[mask_bin]), N_each_bin)
            else:
                picked_p = random.sample(list(p[mask_bin]), np.sum(mask_bin))
                picked_p = SF.fillresize1d(
                    np.array(picked_p), ax0=N_each_bin, dtype="float"
                )

            out[i, :] = picked_p

            i += 1

    out_flat = out.flatten()

    np.random.shuffle(out_flat)

    d, i, j = np.unravel_index(
        out_flat[~np.isnan(out_flat)][: int(N)].astype(int), shape=arr1.shape
    )

    return d, i, j


def random_sample(arr, mask, size):
    """Randomly samples indices from an array of shape = arr.shape"""

    picked_pixels = random.sample(list(np.arange(arr.size)[mask.flatten()]), int(size))

    d, i, j = np.unravel_index(picked_pixels, shape=arr.shape)

    return d, i, j


def main():

    wdir, ifgs, dates, SHP_i, SHP_j, n_siblings, coherence_2015 = importData()

    n_siblings_extend = np.resize(n_siblings, coherence_2015.shape)

    meandiff = SF.meanDiff(abs(coherence_2015), 11)

    mask = (
        abs(meandiff) < (np.nanmean(abs(meandiff)) + 2 * np.nanstd(abs(meandiff)))
    ) & (n_siblings > 14)

    mask[:6] = False
    mask[81:] = False

    f = lambda x, y: (435 * x) / (1949 * 2) + 435 / 2 - y

    z, y, x = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1], 0 : mask.shape[2]]

    mask_train = np.zeros(mask.shape, dtype=bool)
    mask_test = np.zeros(mask.shape, dtype=bool)

    mask_train[(f(x, y) > 0) & (z < 81) & (z > 6) & mask] = True
    mask_test[(f(x, y) < 0) & (z < 81) & (z > 6) & mask] = True

    # plt.pcolormesh(np.sum(mask_train, axis=0))
    # plt.colorbar()

    # plt.figure()
    # plt.pcolormesh(np.sum(mask_test, axis=0) + np.sum(mask_train, axis=0))
    # plt.colorbar()
    # plt.show()

    N = 20000
    ratio_True = 0.5
    ratio_train = 0.75

    suffix, r = "_ratio", True

    if Path(f"df{suffix}.csv").exists():
        df = pd.read_csv(f"df{suffix}.csv")
    else:
        try:
            d1_train, i1_train, j1_train = np.load(f"d1train{suffix}.npy")
            d2_train, i2_train, j2_train = np.load(f"d2train{suffix}.npy")
            d1_test, i1_test, j1_test = np.load(f"d1test{suffix}.npy")
            d2_test, i2_test, j2_test = np.load(f"d1test{suffix}.npy")
            print("Files found. Continuing. ")
        except FileNotFoundError:
            print("File not found. Creating the coords of the test/train set. ")

            d1_train, i1_train, j1_train = sampleUniform2D(
                abs(coherence_2015)[:82],
                n_siblings_extend[:82],
                mask=mask_train[:82],
                N=N * ratio_train * ratio_True,
                bins=15,
            )
            d2_train, i2_train, j2_train = sampleUniform2D(
                abs(coherence_2015)[:82],
                n_siblings_extend[:82],
                mask=mask_train[:82],
                N=N * ratio_train * ratio_True,
                bins=15,
            )

            d1_test, i1_test, j1_test = sampleUniform2D(
                abs(coherence_2015)[:82],
                n_siblings_extend[:82],
                mask=mask_test[:82],
                N=N * (1 - ratio_train) * ratio_True,
                bins=15,
            )
            d2_test, i2_test, j2_test = sampleUniform2D(
                abs(coherence_2015)[:82],
                n_siblings_extend[:82],
                mask=mask_test[:82],
                N=N * (1 - ratio_train) * ratio_True,
                bins=15,
            )

            # d1_train, i1_train, j1_train = random_sample(
            #     coherence_2015[:82],
            #     mask=mask_train[:82],
            #     size=N * ratio_train * ratio_True,
            # )
            # d2_train, i2_train, j2_train = random_sample(
            #     coherence_2015[:82],
            #     mask=mask_train[:82],
            #     size=N * ratio_train * ratio_True,
            # )

            # d1_test, i1_test, j1_test = random_sample(
            #     coherence_2015[:82],
            #     mask=mask_test[:82],
            #     size=N * (1 - ratio_train) * ratio_True,
            # )
            # d2_test, i2_test, j2_test = random_sample(
            #     coherence_2015[:82],
            #     mask=mask_test[:82],
            #     size=N * (1 - ratio_train) * ratio_True,
            # )

            # ===========================

            np.save(f"d1train{suffix}.npy", np.stack((d1_train, i1_train, j1_train)))
            np.save(f"d2train{suffix}.npy", np.stack((d2_train, i2_train, j2_train)))

            np.save(f"d1test{suffix}.npy", np.stack((d1_test, i1_test, j1_test)))
            np.save(f"d2test{suffix}.npy", np.stack((d2_test, i2_test, j2_test)))

        d1 = np.concatenate((d1_train, d1_test))
        i1 = np.concatenate((i1_train, i1_test))
        j1 = np.concatenate((j1_train, j1_test))

        d2 = np.concatenate((d2_train, d2_test))
        i2 = np.concatenate((i2_train, i2_test))
        j2 = np.concatenate((j2_train, j2_test))

        coords1 = np.vstack((i1, j1)).T
        coords2 = np.vstack((i2, j2)).T

        # Goes through all the coords and chooses which ones to go forwards
        sims = SF.generateSims(
            coords1, coords2, d1, d2, ifgs, SHP_i, SHP_j, n_changes=None
        )
        # output:
        # coord1
        # coord2
        # n_siblings
        # std_jackknifed_coherence
        # bias_jackknife
        # mean_amplitude
        # std_amplitude
        # amplitude_of_POI
        # amp_difference_between_current_and_prev_for_POI
        # max_difference_of_all_sibs_for current and previous image
        # coherence

        headers = [
            "i",
            "j",
            "n_siblings",
            "jk_std",
            "jk_bias",
            "amp_mean",
            "amp_std",
            "amp_px",
            "poi_diff",
            "max_amp_diff",
            "actual_coherence",
            "apparent_coherence",
        ]

        df_sims = pd.DataFrame(sims, columns=np.array(headers))

        print(df_sims)  #########################################

        df_sims["apparent_coherence"] = np.concatenate(
            abs(df_sims["apparent_coherence"])
        )
        df_sims["actual_coherence"] = np.concatenate(abs(df_sims["actual_coherence"]))

        df = df_sims.copy()
        labels = abs(df.apparent_coherence - df.actual_coherence) > 0.1
        df["labels"] = labels

        # Now need to add the unchanged values on to it
        n_changed01 = np.sum(
            abs(df_sims.apparent_coherence - df_sims.actual_coherence) > 0.1
        )

        try:
            d3, i3, j3 = np.load(f"d3testtrain{suffix}.npy")
            # i3 = np.load("i3testtrain.npy")
            # j3 = np.load("j3testtrain.npy")

            coords3 = np.vstack((i3, j3)).T
        except FileNotFoundError:
            d3_train, i3_train, j3_train = sampleUniform2D(
                abs(coherence_2015)[:82],
                n_siblings_extend[:82],
                mask=mask_train[:82],
                N=n_changed01 * ratio_train * (1 - ratio_True) * 2,
                bins=15,
            )
            d3_test, i3_test, j3_test = sampleUniform2D(
                abs(coherence_2015)[:82],
                n_siblings_extend[:82],
                mask=mask_test[:82],
                N=n_changed01 * (1 - ratio_train) * (1 - ratio_True) * 2,
                bins=15,
            )

            # d3_train, i3_train, j3_train = random_sample(
            #     coherence_2015[:82],
            #     mask=mask_train[:82],
            #     size=n_changed01 * ratio_train * (1 - ratio_True) * 2,
            # )
            # d3_test, i3_test, j3_test = random_sample(
            #     coherence_2015[:82],
            #     mask=mask_test[:82],
            #     size=n_changed01 * (1 - ratio_train) * (1 - ratio_True) * 2,
            # )

            d3 = np.concatenate((d3_train, d3_test))
            i3 = np.concatenate((i3_train, i3_test))
            j3 = np.concatenate((j3_train, j3_test))

            np.save(f"d3testtrain{suffix}.npy", np.stack((d3, i3, j3)))
            # np.save("i3testtrain.npy", i3)
            # np.save("j3testtrain.npy", j3)

            coords3 = np.vstack((i3, j3)).T

        sims_unchanged = SF.generateSims(
            coords3, coords3, d3, d3, ifgs, SHP_i, SHP_j, n_changes=None
        )

        df_sims_unchanged = pd.DataFrame(sims_unchanged, columns=np.array(headers))

        df_sims_unchanged["apparent_coherence"] = np.concatenate(
            abs(df_sims_unchanged["apparent_coherence"])
        )
        df_sims_unchanged["actual_coherence"] = np.concatenate(
            abs(df_sims_unchanged["actual_coherence"])
        )

        df_sims = df_sims[
            abs(df_sims.apparent_coherence - df_sims.actual_coherence) > 0.1
        ]

        df = pd.concat((df_sims, df_sims_unchanged), ignore_index=True)

        df = get_selection_metrics_ratio(
            "/nfs/a1/homes/py15jmc/bootstrap/2023/Sibling_sel_window_metrics/IFG_all.csv",
            df,
        )

        labels = np.ones(df.shape[0], dtype=bool)
        labels[-sims_unchanged.shape[0] :] = False
        df["labels"] = labels

        print("Headers")
        print(list(df))

        df.to_csv(f"df{suffix}.csv", index=False, header=True)

    plt.figure()

    i_plot, j_plot = np.mgrid[0 : np.max(df["i"]), 0 : np.max(df["j"])]

    split_mask_ = f(j_plot, i_plot) > 0

    plt.pcolormesh(split_mask_, cmap="binary", alpha=0.4)
    plt.scatter(df["j"], df["i"], c=df["labels"], s=2, cmap="bwr")

    # Doing the random forest
    labels = np.array(df["labels"])
    print(labels)

    print(list(df))

    # features = df[["n_siblings", "jk_std", "jk_bias", "amp_mean", "amp_px", "max_amp_diff", "apparent_coherence"]
    features = df[
        [
            "n_siblings",
            "jk_std",
            "jk_bias",
            "amp_mean",
            "max_amp_diff",
            "poi_diff",
            "apparent_coherence",
        ]
    ]

    # print(np.where(np.isnan(df)))
    print(df)
    feature_list = np.array(list(features.columns))

    split_mask = f(df.j, df.i) <= 0

    train_features = features[~split_mask]
    test_features = features[split_mask]

    train_labels = labels[~split_mask]
    test_labels = labels[split_mask]

    SF.test_training_pie(train_labels, test_labels)

    rf, predictions, (fpr, tpr), importance = SF.runRF(
        train_features, train_labels, test_features, test_labels
    )

    # datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    # joblib.dump(rf, f"RF_{datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')}.jbl")
    joblib.dump(rf, f"RF{suffix}.jbl")

    plt.show()


def get_selection_metrics_ratio(filename, df_sims):
    df_metrics = pd.read_csv(filename)

    # Get the indices of the coords used in training.
    i = df_sims["i"].astype(int)
    j = df_sims["j"].astype(int)

    indices = []

    # Loop through the training indeces and find the corresponding mean metrics
    # for the estimation period.
    for i_, j_ in zip(i, j):
        ix = np.where((df_metrics["i"] == i_) & (df_metrics["j"] == j_))[0][0]
        indices.append(ix)

    # Based on the dataframe position indices in the estimation period metrics,
    # retrieve those values and put into the metrics df sims dataframe.
    # df_metrics_sims = df_metrics.loc[df_sims.index[indices]]
    df_metrics_sims = df_metrics.loc[indices]

    # Get the ratio between the df_sims (values for this particular image) and
    # the metrics in the estimation period.

    # df_metrics_sims[df_metrics_sims == 0] = np.nan

    # df_sims[df_sims == 0] = np.nan

    out = df_sims.copy()

    out.reset_index(drop=True, inplace=True)
    df_metrics_sims.reset_index(drop=True, inplace=True)

    # print(f"{out = }")
    # print(f"{df_metrics_sims = }")

    out[list(out)[3:]] = out[list(out)[3:]] / df_metrics_sims[list(df_metrics_sims)[3:]]
    # out = df_sims / df_metrics_sims

    return out


if __name__ == "__main__":
    sys.exit(main())
