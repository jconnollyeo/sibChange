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

SHP_i = np.load(wdir + "bootstrap/shp_i_20150511_20170617.npy")
SHP_j = np.load(wdir + "bootstrap/shp_j_20150511_20170617.npy")

n_siblings = np.sum(~np.isnan(SHP_i) * ~np.isnan(SHP_j), axis=0)

coherence_2015 = np.load(wdir + "bootstrap/coherence_2015.npy")

# ix1, ix2 = 100, 101


def get_selection_metrics_ratio(filename, df_sims):
    df_metrics = pd.read_csv(filename)
    # Based on the dataframe position indices in the estimation period metrics,
    # retrieve those values and put into the metrics df sims dataframe.
    # df_metrics_sims = df_metrics.loc[df_sims.index[indices]]
    # Get the ratio between the df_sims (values for this particular image) and
    # the metrics in the estimation period.
    out = df_sims.copy()
    out.reset_index(drop=True, inplace=True)
    df_metrics.reset_index(drop=True, inplace=True)
    out[list(out)[3:]] = out[list(out)[3:]] / df_metrics[list(df_metrics)[3:]]
    return out


# ixs_ = np.load("ixs.npy")
# ixs = np.asarray([i for i in np.arange(81) if i not in ixs_])
# for ix1, ix2 in zip(np.arange(90, 110), np.arange(91, 111)):
for ix1, ix2 in zip([100], [101]):

    if False:
        pass
    else:
        print(ix1, ix2)
        ifg1 = ifgs[ix1]
        ifg2 = ifgs[ix2]
        suffix = "_median"
        metrics = SF.generateMetricsIFG(ifg1, ifg2, SHP_i, SHP_j)

        # Dealing with inf or nan values in metrics

        df = pd.DataFrame(
            metrics,
            columns=np.array(
                [
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
            ),
        )
        df.to_csv(
            f"/nfs/a1/homes/py15jmc/bootstrap/2023/Sibling_sel_window_metrics/IFG{suffix}_{str(ix1).zfill(3)}_{str(ix2).zfill(3)}.csv"
        )

        df = get_selection_metrics_ratio(
            f"/nfs/a1/homes/py15jmc/bootstrap/2023/Sibling_sel_window_metrics/IFG_all{suffix}.csv",
            df,
        )  # Turn it into ratios

        dropped_mask = np.ones(ifg1.shape, dtype=bool)

        # Remove any nan values
        ix_nan = list(
            set(
                list(set(np.where(np.isnan(df))[0]))
                + list(set(np.where(np.isinf(df))[0]))
            )
        )

        print(f"{len(ix_nan) = }")

        for ix in ix_nan:
            # print (f"Dropping bc nan: {ix}")
            try:
                dropped_mask[int(df["i"][ix]), int(df["j"][ix])] = False
            except:
                print(f"Error : {ix}")

        # Remove any pixels that have fewer than 15 siblings.
        ix_sibs = list(set(np.where(df["n_siblings"] < 15)[0]))
        for ix in ix_sibs:
            # print (f"Dropping bc sibs: {ix}")
            try:
                dropped_mask[int(df["i"][ix]), int(df["j"][ix])] = False
            except:
                print(f"Error : {ix}")
        drop_ix = list(set(ix_nan + ix_sibs))

        print(f"Dropping {len(drop_ix)}")
        for ix in drop_ix:
            df = df.drop(ix)

        np.save(
            f"actual/dropped_mask_{ix1}_{ix2}{suffix}.npy",
            dropped_mask,
        )

        df.to_csv(f"actual/IFG_{ix1}_{ix2}_dropped{suffix}.csv")
        # RF = joblib.load("RF_20230215_121116.jbl")

        RF = joblib.load("RF_ratio.jbl")

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
        predictions = RF.predict(features)

        predictions_arr = np.zeros(ifg1.shape, dtype=bool)

        predictions_arr[dropped_mask] = predictions

        probs_arr = np.zeros(ifg1.shape, dtype=float)

        probs = RF.predict_proba(features)[:, 0]

        probs_arr[dropped_mask] = probs
        np.save(
            f"actual/probabilities_{ix1}_{ix2}{suffix}.npy",
            probs_arr,
        )
        np.save(
            f"actual/predictions_{ix1}_{ix2}{suffix}.npy",
            predictions_arr,
        )

        # pred_arr = predictions.reshape(ifg1.shape)

        # fig, ax = plt.subplots()
        # ax.matshow(np.mean(abs(ifgs), axis=0), cmap="binary_r")
        # ax.matshow(SF.falseNaN(predictions_arr), cmap="RdBu")

        # fig, ax = plt.subplots()
        # mask_siblings, predictions_filt = SF.filterPredictions(predictions_arr, SHP_i, SHP_j, 5)
        # ax.matshow(np.mean(abs(ifgs), axis=0), cmap="binary_r")
        # ax.matshow(SF.falseNaN(predictions_filt), cmap="RdBu")

        # plt.show()
