#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


basePath = "/home/jgonza96/Projects/activeMatterCommunities/workspace/processed/parameterSpaceTable.csv"
df = pd.read_csv(basePath)


#%%
for name in ["edge_frac", "H_1", "H_2", "nematic", "r_max", "cellnumber", "time"]:
    gdf = df.groupby(["max_aspect_ratio_1", "max_aspect_ratio_2"])
    gdf = gdf.agg({name: "mean"}).reset_index()

    fig, ax = plt.subplots(figsize=(8,6))

    pivoted = gdf.pivot_table(index="max_aspect_ratio_1", columns="max_aspect_ratio_2", values=name)
    sns.heatmap(pivoted, cbar_kws={"label": name})

    ax.set_xlabel("Division Length / thickness")
    ax.set_ylabel("Division Length / thickness")


#%%
for x in df.rng_seed:
    print(x)

# %%
# select the fourth row of datafame df
df.iloc[4]