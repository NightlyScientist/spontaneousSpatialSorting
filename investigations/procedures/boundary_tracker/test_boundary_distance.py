# %%
import os
import glob
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from collections import defaultdict

path_ = "R:1_MP:20000_FS:5000_S:50.0_IT:passive_ring_IC:50_D:0.0_K:100.0_dT:0.001_T:0.1,0.1_MAR:3.0,2.0_GR:0.02,0.02_EDT:0_EA:0_AF:1.0_CC:100.0_RA:0.0/"


def read_periphery(periphery_csv):
    df = pd.read_csv(periphery_csv)
    points = df[["x", "y"]].values
    return Polygon(points)


def create_rod_capsule(center, orientation, length, radius, resolution=8):
    """
    Create a Shapely Polygon representing a rod as a capsule (rectangle + semicircular ends).
    """
    orientation = orientation / np.linalg.norm(orientation)
    normal = np.array([-orientation[1], orientation[0]])

    # Endpoints of the centerline
    p1 = center + 0.5 * length * orientation
    p2 = center - 0.5 * length * orientation

    # Rectangle corners
    corner1 = p1 + radius * normal
    corner2 = p1 - radius * normal
    corner3 = p2 - radius * normal
    corner4 = p2 + radius * normal

    # Arc points for the caps
    angles1 = np.linspace(0, np.pi, resolution)
    angles2 = np.linspace(np.pi, 2 * np.pi, resolution)

    # Rotation matrix for orientation
    rot = np.array(
        [[orientation[0], -orientation[1]], [orientation[1], orientation[0]]]
    )

    arc1 = [p1 + radius * (rot @ np.array([np.cos(a), np.sin(a)])) for a in angles1]
    arc2 = [p2 + radius * (rot @ np.array([np.cos(a), np.sin(a)])) for a in angles2]

    # Full outline (ordered)
    outline = [corner1] + arc1 + [corner2, corner3] + arc2 + [corner4]
    return Polygon(outline)

def signed_centerdistance_to_periphery(rod_df, periphery_polygon, resolution=8):
    """
    Compute the minimum distance from each rod's center (as a point) to the nearest periphery boundary.
    Returns a numpy array of signed distances
    """
    distances = []
    for _, row in rod_df.iterrows():
        p_x = np.asarray(periphery_polygon.exterior.xy[0])
        p_y = np.asarray(periphery_polygon.exterior.xy[1])
        dists_ = np.hypot(row["x"] - p_x, row["y"] - p_y)
        min_dist_index = np.argmin(dists_)
        dist = dists_[min_dist_index]

        # Negative if inside, positive if outside
        center = Point(row["x"], row["y"])
        if periphery_polygon.contains(center):
            dist = -dist
        distances.append(dist)
    return np.array(distances)

def signed_distance_to_periphery(rod_df, periphery_polygon, resolution=8):
    """
    Compute the signed minimum distance from each rod's body (capsule) to the periphery polygon.
    Negative if inside, positive if outside.
    """
    distances = []
    for _, row in rod_df.iterrows():
        center = np.array([row["x"], row["y"]])
        orientation = np.array([row["ex"], row["ey"]])
        length = row["l"]
        radius = row["cell_r"]
        rod_shape = create_rod_capsule(center, orientation, length, radius, resolution)
        dist = rod_shape.distance(periphery_polygon)
        # Negative if inside, positive if outside
        if periphery_polygon.contains(rod_shape):
            dist = -dist
        distances.append(dist)
    return np.array(distances)


def extract_timestep(filename):
    base = os.path.basename(filename)
    parts = base.replace(".csv", "").split("_")
    return int(parts[-1]) if parts[-1].isdigit() else None


def process_timestep(args):
    rod_file, periphery_file, visualize, outdir = args
    timestep = extract_timestep(rod_file)

    df = pd.read_csv(rod_file)
    if "cell_r" not in df.columns:
        raise ValueError("Rod CSV must include a 'radius' column for thickness.")

    df_label1 = df[df["color2"] == 1].copy()

    if df_label1.empty:
        return None

    periphery_polygon = read_periphery(periphery_file)
    distances = signed_centerdistance_to_periphery(df_label1, periphery_polygon)

    df_label1["signed_distance_to_periphery"] = distances
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    n = len(distances)

    # Visualization
    if visualize:
        plt.figure(figsize=(7, 7))
        # Periphery
        x, y = periphery_polygon.exterior.xy
        plt.plot(x, y, "k-", lw=2, label="Periphery")
        #plt.scatter(x, y, label="Periphery")
        # Rods, color-coded by signed distance
        sc = plt.scatter(
            df_label1["x"],
            df_label1["y"],
            c=distances,
            cmap="coolwarm",
            s=60,
            edgecolor="k",
            vmin=-max(abs(distances)),
            vmax=max(abs(distances)),
        )
        plt.colorbar(sc, label="Signed distance to periphery")
        plt.title(f"Timestep {timestep} : d = {mean_dist:.2f} ± {std_dist:.2f} (n={n})")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(
        #    os.path.join(outdir, f"periphery_distances_{timestep}.png"), dpi=200
        # )
        # plt.close()

    # Save per-rod distances
    # df_label1.to_csv(
    #    os.path.join(outdir, f"label1_signed_distances_{timestep}.csv"), index=False
    # )
    return {
        "timestep": timestep,
        "mean_signed_distance": mean_dist,
        "std_signed_distance": std_dist,
        "n_rods": n,
    }


def main(rod_dir, periphery_dir, output_csv, outdir, visualize=True, max_workers=None):
    rod_files = np.array([
        f
        for f in glob.glob(os.path.join(rod_dir, "*.csv"))
        if os.path.basename(f) != "opts.csv"
    ])
    times_stamps_sorter = np.argsort(np.array([extract_timestep(rod_file) for rod_file in rod_files]))

    args_list = []
    for rod_file in rod_files[times_stamps_sorter]:
        timestep = extract_timestep(rod_file)
        if timestep is None:
            continue
        periphery_file = os.path.join(periphery_dir, f"periphery_{timestep}.csv")
        if not os.path.exists(periphery_file):
            print(f"Skipping timestep {timestep}: no periphery file found.")
            continue
        args_list.append((rod_file, periphery_file, visualize, outdir))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(
            executor.map(process_timestep, args_list),
            total=len(args_list),
            desc="Timesteps",
        ):
            if result:
                results.append(result)

    # Save time series
    df_results = pd.DataFrame(results).sort_values("timestep")
    # df_results.to_csv(output_csv, index=False)
    # print(f"\nSaved time series to {output_csv}")

    # opts_file = os.path.join(rod_dir, "opts.csv")
    # shutil.copy2(opts_file, output_csv.replace(".csv", "_opts.csv"))
    return df_results


# %%
# -------------------------
# Example usage
# -------------------------
output_csv = "signed_distance_timeseries_capsule.csv"
outdir = "./periphery_viz_out/"

csv_folder = "/home/jgonzaleznunez/Projects/spontaneousSpatialSorting/workspace/data/data/archived/floating_particles_2"
periphery_dir = "./periphery_out/"  # Where to save periphery CSVs and plots
timeseries_dir = "./timeseries_out/"  # Where to save timeseries CSVs

method = "alpha"  # 'alpha' or 'convex'
alpha = 0.2  # Alpha value for alpha shape
save_plot = True  # Save visualization
max_workers = None  # Use all available cores
partcle_label_exclude = 1

#sims = [path_]
sims = os.listdir(periphery_dir)

timeseries_collection = dict()
for sim in sims:
    sim_csv_folder = os.path.join(csv_folder, sim)
    sim_periphery_folder = os.path.join(periphery_dir, sim)
    sim_outdir = os.path.join(outdir, sim)
    timeseries_csv = os.path.join(timeseries_dir, sim, output_csv)

    timeseries = main(
        sim_csv_folder,
        sim_periphery_folder,
        timeseries_csv,
        sim_outdir,
        visualize=False,
        max_workers=None,
    )
    timeseries_collection[sim] = timeseries


# %%
# Remove the seed (e.g., 'R:1_') from sim names and combine timeseries by the rest of the key

def strip_seed(sim_name):
    parts = sim_name.split("_", 1)
    if parts[0].startswith("R:") and len(parts) > 1:
        return parts[1]
    return sim_name

def strip_seed_and_combine_dictionary(timeseries_collection):
    combined_timeseries = defaultdict(list)
    for sim, timeseries in timeseries_collection.items():
        key = strip_seed(sim)
        combined_timeseries[key].append([sim, timeseries])

    # Optionally, concatenate timeseries for each key
    #for key in combined_timeseries:
        #combined_timeseries[key] = pd.concat(combined_timeseries[key], ignore_index=True)
    return combined_timeseries

# %%
fig, ax = plt.subplots(figsize=(10, 6))

combined_timeseries = strip_seed_and_combine_dictionary(timeseries_collection)
finel_point_table = []

for timeseries_subset in combined_timeseries.values():
    # Collect all timesteps and mean_signed_distance arrays
    all_timesteps = []
    all_means = []

    for fname, ts in timeseries_subset:
        if ts.timestep.max() > 100000:
            all_timesteps.append(ts["timestep"].values)
            all_means.append(ts["mean_signed_distance"].values)

    if len(all_timesteps) == 0:
        continue

    ## get the opts file for sim collection
    fname = timeseries_subset[0][0]
    opts_file = os.path.join(csv_folder, fname, "opts.csv")
    opts = pd.read_csv(opts_file).to_dict(orient="index")[0]

    # Find common timesteps (intersection)
    common_timesteps = set(all_timesteps[0])
    for t in all_timesteps[1:]:
        common_timesteps &= set(t)
    common_timesteps = sorted(list(common_timesteps))

    # Stack mean_signed_distance for common timesteps
    stacked_means = []
    for i in range(len(all_means)):
        idx = np.isin(all_timesteps[i], common_timesteps)
        stacked_means.append(all_means[i][idx])
    stacked_means = np.stack(stacked_means, axis=0)

    mean_curve = np.mean(stacked_means, axis=0)
    std_curve = np.std(stacked_means, axis=0) / np.sqrt(len(stacked_means))

    common_timesteps = np.array(common_timesteps) * opts["time_step"]

    ax.plot(common_timesteps, mean_curve, label="Mean")
    ax.fill_between(
        common_timesteps,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.3,
        label="±1 std",
    )

    finel_point_table.append([opts, mean_curve[-1], std_curve[-1], len(stacked_means)])

ax.legend()
#ax.set_xlim(75000, 185000)

# %%
plt.rcParams.update({"font.size": 16})
fig, ax = plt.subplots(figsize=(10, 6))

for opts_, mean, std, n in finel_point_table:
    l_1, l_2 = opts_["division_length_1"], opts_["division_length_2"]
    d_1, d_2 = opts_["thickness_1"], opts_["thickness_2"]
    A_1, A_2 = l_1 * d_1, l_2 * d_2
    aspect_ratio = l_2 * d_1 / l_1 / d_2

    color = "blue" if l_2 / d_2 < 4 else "black"
    label = rf"$a_2$ = {l_2 / d_2}"

    ax.errorbar(
        #aspect_ratio,
        A_2 / A_1,
        mean / d_2,
        yerr=std / d_2 ,
        fmt="o",
        color=color,
        label = label,
    )
    #ax.axhline(y=0, color="black", linestyle="--", linewidth=0.1)

xticks = list(ax.get_xticks())
if 1 not in xticks:
    xticks.append(1)
    xticks = sorted(xticks)
ax.set_xticks(xticks)
ax.set_xlim(-2, 30)

ax.set_xlabel(r"Area ratio ($A_2/A_1$)")
ax.set_ylabel(r"Mean signed distance / $d_2$")

# Remove duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc="lower right", fontsize=14)