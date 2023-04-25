import glob
import pandas as pd

# Define the pattern of the file names
file_pattern = "IFG_*_*.csv"

# Get a list of all the file names that match the pattern
file_list = glob.glob(file_pattern)

file_list.sort()

with pd.read_csv(file_list[0]) as df:
    headers = list(df)
    number_of_features = len(headers)
    max_i = df["i"].max()
    max_j = df["j"].max()

store_metrics = np.empty((number_of_features, max_i, max_j))

for filename in file_list:
    df = pd.read_csv(filename)
    for ix, feature in zip(list(df)):
        store_metrics[ix, int(df["i"]), int(df["j"])] += df[feature]

df_out = pd.DataFrame(columns=headers)
for x, header in enumerate(headers):
    df_out[header] = store_metrics[x].flatten() / len(file_list)

# np.save("mean_metrics.npy", store_metrics/len(file_list))
df_out.to_csv(
    f"/nfs/a1/homes/py15jmc/bootstrap/2023/Sibling_sel_window_metrics/IFG_all.csv"
)
