# %%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import MultiPoint
from shapely.ops import unary_union, polygonize
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import shutil

# ----------------------------
# Utility Functions
# ----------------------------


def read_rod_csv(filename):
    """Read rod data from CSV. Assumes columns: pos_x, pos_y, ori_x, ori_y, length"""
    df = pd.read_csv(filename)
    return df


def rod_outline_points(df, resolution=4):
    """Compute outline points for rods with thickness."""
    centers = df[["x", "y"]].values
    oris = df[["ex", "ey"]].values
    lengths = df["l"].values
    radii = df["cell_r"].values

    points = []

    for i in range(len(df)):
        c = centers[i]
        o = oris[i]
        r = radii[i]
        l = lengths[i]
        n = np.array([-o[1], o[0]])  # Normal vector

        p1 = c + 0.5 * l * o
        p2 = c - 0.5 * l * o

        # Rectangle corners
        p1_left = p1 + r * n
        p1_right = p1 - r * n
        p2_left = p2 + r * n
        p2_right = p2 - r * n

        # Optionally add arc points for rounded ends
        arc1 = [
            p1 + r * np.array([np.cos(a), np.sin(a)])
            for a in np.linspace(
                np.arctan2(n[1], n[0]), np.arctan2(-n[1], -n[0]), resolution
            )
        ]
        arc2 = [
            p2 + r * np.array([np.cos(a), np.sin(a)])
            for a in np.linspace(
                np.arctan2(-n[1], -n[0]), np.arctan2(n[1], n[0]), resolution
            )
        ]

        # Combine all outline points
        rod_pts = [p1_left, p2_left] + arc2 + [p2_right, p1_right] + arc1
        points.extend(rod_pts)

    return np.array(points)


def compute_convex_hull(endpoints):
    """Compute convex hull of endpoints."""
    if len(endpoints) < 3:
        return endpoints
    hull = ConvexHull(endpoints)
    return endpoints[hull.vertices]


def compute_alpha_shape(points, alpha=0.2):
    """Compute alpha shape (concave hull) of a set of points."""
    if len(points) < 4:
        return points

    def add_edge(edges, i, j):
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    for ia, simplex in enumerate(tri.simplices):
        pts = points[simplex]
        a = np.linalg.norm(pts[0] - pts[1])
        b = np.linalg.norm(pts[1] - pts[2])
        c = np.linalg.norm(pts[2] - pts[0])
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        if area == 0:
            continue
        circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / alpha:
            add_edge(edges, simplex[0], simplex[1])
            add_edge(edges, simplex[1], simplex[2])
            add_edge(edges, simplex[2], simplex[0])

    edge_points = [points[list(edge)] for edge in edges]
    m = MultiPoint(points)
    polygons = list(polygonize(edge_points))
    if not polygons:
        return compute_convex_hull(points)
    union = unary_union(polygons)

    if union.geom_type == "Polygon":
        return np.array(union.exterior.coords)
    elif union.geom_type == "MultiPolygon" and hasattr(union, "geoms"):
        # Take the largest polygon by area
        largest = max(union.geoms, key=lambda p: p.area)
        return np.array(largest.exterior.coords)
    else:
        return compute_convex_hull(points)


def compute_periphery(endpoints, method="alpha", alpha=0.2):
    if method == "convex":
        return compute_convex_hull(endpoints)
    elif method == "alpha":
        return compute_alpha_shape(endpoints, alpha)
    else:
        raise ValueError("Unknown method: choose 'alpha' or 'convex'")


def plot_periphery(endpoints, periphery, title=None, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(endpoints[:, 0], endpoints[:, 1], s=2, label="Rod Endpoints", alpha=0.5)
    plt.plot(periphery[:, 0], periphery[:, 1], "r-", lw=2, label="Periphery")
    plt.axis("equal")
    plt.legend()
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


# ----------------------------
# Parallel Processing Function
# ----------------------------


def process_file(
    filename,
    output_dir,
    opts_file,
    method="alpha",
    alpha=0.2,
    save_plot=True,
    partcle_label_exclude=None,
):
    try:
        df = read_rod_csv(filename)
        if partcle_label_exclude is not None:
            df = df[df["color2"] != partcle_label_exclude]

        shutil.copy2(opts_file, os.path.join(output_dir, "opts.csv"))

        points = rod_outline_points(df)
        periphery = compute_periphery(points, method=method, alpha=0.2)

        # Save periphery to CSV
        timestep = os.path.splitext(os.path.basename(filename))[0].split("_")[-1]
        out_csv = os.path.join(output_dir, f"periphery_{timestep}.csv")
        pd.DataFrame(periphery, columns=["x", "y"]).to_csv(out_csv, index=False)

        # Save plot
        if save_plot:
            out_plot = os.path.join(output_dir, f"periphery_{timestep}.png")
            plot_periphery(
                points, periphery, title=f"Timestep {timestep}", save_path=out_plot
            )

        return timestep, len(periphery)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


# ----------------------------
# Main Batch Processing
# ----------------------------


def process_opts(csv_path):
    opts = pd.read_csv(csv_path, header=0).to_dict(orient="index")[0]
    return opts


def batch_process(
    csv_dir,
    output_dir,
    method="alpha",
    alpha=0.2,
    save_plot=True,
    max_workers=None,
    partcle_label_exclude=None,
):
    opts_file = os.path.join(csv_dir, "opts.csv")

    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(
        f
        for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if os.path.basename(f) != "opts.csv"
    )

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_file,
                f,
                output_dir,
                opts_file,
                method,
                alpha,
                save_plot,
                partcle_label_exclude,
            )
            for f in csv_files
        ]
        for future in tqdm(futures, desc="Processing timesteps"):
            result = future.result()
            if result:
                results.append(result)

    print(f"Processed {len(results)} timepoints.")
    return results


# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    csv_folder = "/home/jgonzaleznunez/Projects/spontaneousSpatialSorting/workspace/data/floating_particles_2/"
    output_folder = "./periphery_out/"  # Where to save periphery CSVs and plots
    method = "alpha"  # 'alpha' or 'convex'
    alpha = 0.2  # Alpha value for alpha shape
    save_plot = True  # Save visualization
    max_workers = None  # Use all available cores
    partcle_label_exclude = 1

    sims = os.listdir(csv_folder)
    # Run batch processing
    for sim in sims:
        sim_csv_folder = os.path.join(csv_folder, sim)
        sim_output_folder = os.path.join(output_folder, sim)

        batch_process(
            sim_csv_folder,
            sim_output_folder,
            method,
            alpha,
            save_plot,
            max_workers,
            partcle_label_exclude,
        )
