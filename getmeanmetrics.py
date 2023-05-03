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

store_metrics = np.empty((number_of_features, int(max_i) + 1, int(max_j) + 1))

for filename in tqdm(file_list):
    df = pd.read_csv(filename)
    for ix, feature in enumerate(list(df)):
        store_metrics[ix, df["i"].astype(int), df["j"].astype(int)] += df[feature]

df_out = pd.DataFrame(columns=headers)
for x, header in enumerate(headers):
    df_out[header] = store_metrics[x].flatten() / len(file_list)

np.save("mean_metrics.npy", store_metrics / len(file_list))
df_out.to_csv(
    f"/nfs/a1/homes/py15jmc/bootstrap/2023/Sibling_sel_window_metrics/IFG_all.csv",
    index=False,
)
