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
import math
from collections import namedtuple
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm


def read_rod_csv(filename):
    """Read rod data from CSV. Assumes columns: pos_x, pos_y, ori_x, ori_y, length"""
    df = pd.read_csv(filename)
    return df

def radial_order(x, y, ex, ey):
    for i in range(np.size(x)):
        theta = math.atan2(ey[i], ex[i])
        phi = math.atan2(y[i], x[i])
        yield np.cos(2 * (theta - phi))

def compute_radial_order(df):
    """Compute radial order parameter for each rod."""
    x = df["x"].values
    y = df["y"].values
    ex = df["ex"].values
    ey = df["ey"].values
    #cell_type = df["color2"]
    #c_1 = cell_type == 1
    #c_2 = cell_type == 2

    #sr_1 = np.array(list(radial_order(x[c_1], y[c_1], ex[c_1], ey[c_1])))
    #sr_2 =  np.array(list(radial_order(x[c_2], y[c_2], ex[c_2], ey[c_2])))
    sr =  np.array(list(radial_order(x, y, ex, ey)))
    return np.mean(sr)
    #return sr_1, sr_2, sr

def process_file(
    filename,
    output_dir,
    opts_file,
    partcle_label_exclude=None,
):
    try:
        df = read_rod_csv(filename)
        if partcle_label_exclude is not None:
            df = df[df["color2"] != partcle_label_exclude]

        shutil.copy2(opts_file, os.path.join(output_dir, "opts.csv"))


        # Save periphery to CSV
        timestep = os.path.splitext(os.path.basename(filename))[0].split("_")[-1]
        out_csv = os.path.join(output_dir, f"periphery_{timestep}.csv")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


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
        (f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if os.path.basename(f) != "opts.csv"),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
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

# %%

if __name__ == "__main__":
    csv_folder = "/anisotropic_friction"
    save_plot = True  # Save visualization
    max_workers = None  # Use all available cores
    partcle_label_exclude = 1

    sims = os.listdir(csv_folder)
    # Run batch processing
    for sim in sims:
        sim_csv_folder = os.path.join(csv_folder, sim)

        #batch_process(
        #    sim_csv_folder,
        #    sim_output_folder,
        #    save_plot,
        #    max_workers,
        #    partcle_label_exclude,
        #)

def opts_dir_to_df(csv_folder):
    sims = os.listdir(csv_folder)
    opts_list = []
    for sim in sims:
        csv_dir = os.path.join(csv_folder, sim)
        opts_file = os.path.join(csv_dir, "opts.csv")
        opts_df = pd.read_csv(opts_file, header=0).to_dict(orient="index")[0]
        opts_df["simulation"] = csv_dir
        opts_list.append(opts_df)
    return pd.DataFrame(opts_list)

def sorted_records(csv_dir):
    csv_files = sorted(
        (f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if os.path.basename(f) != "opts.csv"),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )
    return csv_files

def mean_radial_order(csv_dir):
    t, S = [], []
    for filename in sorted_records(csv_dir):
        timestep = os.path.splitext(os.path.basename(filename))[0].split("_")[-1]

        df = read_rod_csv(filename)
        sr = compute_radial_order(df)
        t.append(float(timestep))
        S.append(sr)
    return np.array(t), np.array(S)


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


def compute_periphery_fraction(centers, ctypes, periphery, t, area_1, area_2):
    if centers is None or len(centers) == 0 or periphery is None or len(periphery) < 2:
        return np.array([], dtype=int)

    centers = np.asarray(centers, dtype=float)
    periphery = np.asarray(periphery, dtype=float)

    # Ensure polyline is closed
    if not np.allclose(periphery[0], periphery[-1]):
        periphery = np.vstack([periphery, periphery[0]])

    p0 = periphery[:-1]  # segment start
    p1 = periphery[1:]   # segment end
    seg = p1 - p0
    seg_len2 = np.einsum("ij,ij->i", seg, seg)

    idx_within = []
    for i, c in enumerate(centers):
        w = c - p0
        # Projection parameter on each segment
        u = np.divide(
            np.einsum("ij,ij->i", w, seg),
            seg_len2,
            out=np.zeros_like(seg_len2),
            where=seg_len2 > 0,
        )
        u = np.clip(u, 0.0, 1.0)

        proj = p0 + u[:, None] * seg
        d2 = np.einsum("ij,ij->i", c - proj, c - proj)

        if np.min(d2) <= t * t:
            idx_within.append(i)

    indices = np.asarray(idx_within, dtype=int)
    if len(indices) == 0:
        return 0.0
    c_1 = ctypes[indices] == 0
    c_2 = ctypes[indices] == 1
    A_1 = np.sum(c_1) * area_1
    A_2 = np.sum(c_2) * area_2
    return A_1 / (A_1 + A_2), indices
    #return np.sum(c_1) / (np.sum(c_1) + np.sum(c_2)), indices

# %%
csv_folder = "anisotropic_friction"
save_plot = True  # Save visualization
max_workers = None  # Use all available cores
partcle_label_exclude = 1

odf = opts_dir_to_df(csv_folder).groupby('anisotropic_friction')
A = odf.anisotropic_friction.unique()

norm = Normalize(vmin=0, vmax=len(A)-1)
#norm = Normalize(vmin=A.min(), vmax=A.max())
cmap = cm.get_cmap('managua')

opts_df = opts_dir_to_df(csv_folder).groupby('anisotropic_friction', sort=True)
#for name, group in opts_df:
#    OptsRecord = namedtuple('OptsRecord', group.index)
#    opts_record = OptsRecord(*group.values)


X, Y, C, L = [], [], [], []
Yerr = []
for gidx, (name, group) in enumerate(opts_df):
    print(f"Processing group: {name}")
    t_all, S_all = [], []
    t_div_all = []
    t_step = 1

    for idx, row in group.iterrows():
        sim_dir = row.simulation
        t_step = row.time_step
        t, S = mean_radial_order(sim_dir)
        t_all.append(t)
        S_all.append(S)
        t_div = (row.division_time_1 + row.division_time_2)/2
        t_div_all.append(t_div)
    
    # Align time arrays and compute mean and uncertainty
    t_common = t_all[0]
    S_array = np.array([S for S in S_all])
    S_mean = np.mean(S_array, axis=0)
    S_std = np.std(S_array, axis=0)
    S_sem = S_std / np.sqrt(len(S_all))  # Standard error of the mean
    t_g = np.mean(t_div_all)
    c = cmap(norm(gidx))
    C.append(c)
    X.append(t_common / t_g * t_step)
    Y.append(S_mean)
    Yerr.append(S_sem)
    L.append(f"{name}")
    #c = cmap(norm(name))

# %%

odf = opts_dir_to_df(csv_folder)
# get time in units of doubling time
t_g = (
    odf.division_time_1[-1] / odf.time_step[-1]
)  # use the reference population division time as the generation time scaling

fig, ax = plt.subplots()
print(C)
for t_common, S_mean, S_sem, name, c in zip(X, Y, Yerr, L, C):
    ax.plot(t_common / t_g, S_mean, label=f"{name}", color=c)
    ax.fill_between(t_common * t_g, S_mean - S_sem, S_mean + S_sem, alpha=0.3, color=c)

ax.set_xlabel("$t / T$", fontsize = 18)
ax.set_ylabel(r"$\phi_B$", fontsize=24)
ax.tick_params(axis='both', labelsize=16)
ax.set_xlabel(r"$t / T$", fontsize=18)
ax.set_ylabel(r"$\bar{s}_r$", fontsize=24)
plt.tight_layout()

# %%
opts_df = opts_dir_to_df(csv_folder)
first_row = opts_df.iloc[0]
OptsRow = namedtuple("OptsRow", first_row.index.tolist())
opts_row = OptsRow(*first_row.values.tolist())

csv_dir = opts_row.simulation
filename = sorted_records(csv_dir)[-1]

df = read_rod_csv(filename)

centers = df[["x", "y"]].values
ctypes = df["color2"].values

points = rod_outline_points(df)
periphery = compute_periphery(points, method="alpha", alpha=0.2)

# %%
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    centers[:, 0],
    centers[:, 1],
    c=ctypes,
    cmap="tab10",
    s=10,
    alpha=0.8,
    label="Centers",
)

if periphery is not None and len(periphery) > 1:
    peri = np.asarray(periphery)
    if not np.allclose(peri[0], peri[-1]):
        peri = np.vstack([peri, peri[0]])
    ax.plot(peri[:, 0], peri[:, 1], color="black", lw=2, label="Periphery")

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("y", fontsize=20)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# peri_df = pd.DataFrame(periphery, columns=["x", "y"])
threshold = 1.2
area_1 = opts_row.areas_1
area_2 = opts_row.areas_2
peri_frac, indices = compute_periphery_fraction(centers, ctypes, periphery, threshold, area_1, area_2)
peri_frac
ax.scatter(centers[indices, 0], centers[indices, 1], c="red",s=10, alpha=0.2)
fig
# %%
opts_row