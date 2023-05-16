import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# Define the pattern of the file names
file_pattern = "Sibling_sel_window_metrics/IFG_*_*.csv"

# Get a list of all the file names that match the pattern
file_list = glob.glob(file_pattern)

file_list.sort()

file_list = [
    f"Sibling_sel_window_metrics/IFG_{str(int(i)).zfill(3)}_{str(int(i+1)).zfill(3)}.csv"
    for i in np.arange(81)
]

df = pd.read_csv(file_list[0])
headers = list(df)
number_of_features = len(headers)
max_i = df["i"].max()
max_j = df["j"].max()

store_metrics = np.empty(
    (len(file_list), number_of_features, int(max_i) + 1, int(max_j) + 1)
)


for ix_filename, filename in tqdm(enumerate(file_list)):
    df = pd.read_csv(filename)
    for ix, feature in enumerate(list(df)):
        store_metrics[ix_filename, ix, df["i"].astype(int), df["j"].astype(int)] = df[
            feature
        ]

df_out = pd.DataFrame(columns=headers)
# for x, header in enumerate(headers):
#     df_out[header] = store_metrics[x].flatten() / len(file_list)
median_metrics = np.median(store_metrics, axis=0)
np.save("median_metrics.npy", median_metrics)

arr_out = np.empty((len(headers), median_metrics[0].size))
for i, array in enumerate(median_metrics):
    arr_out[i] = median_metrics[i].flatten()

df_out = pd.DataFrame(arr_out.T, columns=headers)

df_out.to_csv(
    f"/nfs/a1/homes/py15jmc/bootstrap/2023/Sibling_sel_window_metrics/IFG_all_median.csv",
    index=False,
)
