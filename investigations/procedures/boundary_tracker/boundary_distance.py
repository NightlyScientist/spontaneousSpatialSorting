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
    print(f"Processing timestep {timestep} with {len(df_label1)} rods.")
    if df_label1.empty:
        return None
    periphery_polygon = read_periphery(periphery_file)
    distances = signed_distance_to_periphery(df_label1, periphery_polygon)
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
        plt.title(f"Timestep {timestep}")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"periphery_distances_{timestep}.png"), dpi=200
        )
        plt.close()

    # Save per-rod distances
    df_label1.to_csv(
        os.path.join(outdir, f"label1_signed_distances_{timestep}.csv"), index=False
    )
    return {
        "timestep": timestep,
        "mean_signed_distance": mean_dist,
        "std_signed_distance": std_dist,
        "n_rods": n,
    }


def main(rod_dir, periphery_dir, output_csv, outdir, visualize=True, max_workers=None):
    os.makedirs(outdir, exist_ok=True)
    rod_files = sorted(
        f
        for f in glob.glob(os.path.join(rod_dir, "*.csv"))
        if os.path.basename(f) != "opts.csv"
    )
    args_list = []
    for rod_file in rod_files:
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
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved time series to {output_csv}")
    print(df_results.head())

    opts_file = os.path.join(rod_dir, "opts.csv")
    shutil.copy2(opts_file, output_csv.replace(".csv", "_opts.csv"))

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    output_csv = "signed_distance_timeseries_capsule.csv"
    outdir = "./periphery_viz_out/"

    csv_folder = "/home/jgonzaleznunez/Projects/spontaneousSpatialSorting/workspace/data/floating_particles_2/"
    periphery_dir = "./periphery_out/"  # Where to save periphery CSVs and plots
    timeseries_dir = "./timeseries_out/"  # Where to save timeseries CSVs

    method = "alpha"  # 'alpha' or 'convex'
    alpha = 0.2  # Alpha value for alpha shape
    save_plot = True  # Save visualization
    max_workers = None  # Use all available cores
    partcle_label_exclude = 1

    sims = os.listdir(periphery_dir)

    for sim in sims:
        sim_csv_folder = os.path.join(csv_folder, sim)
        sim_periphery_folder = os.path.join(periphery_dir, sim)
        sim_outdir = os.path.join(outdir, sim)
        timeseries_csv = os.path.join(timeseries_dir, sim, output_csv)
        os.makedirs(os.path.join(timeseries_dir, sim), exist_ok=True)

        main(
            sim_csv_folder,
            sim_periphery_folder,
            timeseries_csv,
            sim_outdir,
            visualize=True,
            max_workers=None,
        )

# %%
