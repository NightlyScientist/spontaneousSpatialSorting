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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm


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


def compute_periphery_fraction(centers, ctypes, periphery, t, area_1, area_2):
    if centers is None or len(centers) == 0 or periphery is None or len(periphery) < 2:
        return np.array([], dtype=int)

    centers = np.asarray(centers, dtype=float)
    periphery = np.asarray(periphery, dtype=float)

    # Ensure polyline is closed
    if not np.allclose(periphery[0], periphery[-1]):
        periphery = np.vstack([periphery, periphery[0]])

    p0 = periphery[:-1]  # segment start
    p1 = periphery[1:]  # segment end
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
    return A_1 / (A_1 + A_2)
    return np.sum(c_1) / (np.sum(c_1) + np.sum(c_2))


def sorted_records(csv_dir):
    csv_files = sorted(
        (
            f
            for f in glob.glob(os.path.join(csv_dir, "*.csv"))
            if os.path.basename(f) != "opts.csv"
        ),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0]),
    )
    return csv_files


def process_file(
    filename,
    method="alpha",
    alpha=0.2,
    partcle_label_exclude=None,
    threshold=1.0,
    area_1=1.0,
    area_2=1.0,
):
    try:
        df = read_rod_csv(filename)
        centers = df[["x", "y"]].values
        ctypes = df["color2"].values

        if partcle_label_exclude is not None:
            df = df[df["color2"] != partcle_label_exclude]

        points = rod_outline_points(df)
        periphery = compute_periphery(points, method=method, alpha=alpha)

        # Save periphery to CSV
        timestep = os.path.splitext(os.path.basename(filename))[0].split("_")[-1]
        # peri_df = pd.DataFrame(periphery, columns=["x", "y"])
        peri_frac = compute_periphery_fraction(
            centers, ctypes, periphery, threshold, area_1, area_2
        )

        return timestep, peri_frac
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def process_opts(csv_path):
    opts = pd.read_csv(csv_path, header=0).to_dict(orient="index")[0]
    return opts


def batch_process(
    csv_dir,
    method="alpha",
    alpha=0.2,
    max_workers=None,
    partcle_label_exclude=None,
    threshold=1.0,
):
    opts_file = os.path.join(csv_dir, "opts.csv")
    opts = process_opts(opts_file)
    area_1 = opts.get("areas_1", 1.0)
    area_2 = opts.get("areas_2", 1.0)

    csv_files = sorted_records(csv_dir)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_file,
                f,
                method,
                alpha,
                partcle_label_exclude,
                area_1=area_1,
                area_2=area_2, 
                threshold=threshold,
            )
            for f in csv_files
        ]
        for future in tqdm(futures, desc="Processing timesteps"):
            result = future.result()
            if result:
                results.append(result)

    print(f"Processed {len(results)} timepoints.")
    return results

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

# %%
if __name__ == "__main__":
    csv_folder = "anisotropic_friction"
    method = "alpha"  # 'alpha' or 'convex'
    alpha = 1.2  # Alpha value for alpha shape
    save_plot = True  # Save visualization
    max_workers = None  # Use all available cores
    partcle_label_exclude = None

    X, Y, C, L = [], [], [], []
    Yerr =[]
    opts_df = opts_dir_to_df(csv_folder).groupby('anisotropic_friction', sort=True)

    A = opts_df.anisotropic_friction.unique()
    norm = Normalize(vmin=0, vmax=len(A)-1)
    cmap = cm.get_cmap('managua')

    for gidx, (name, group) in enumerate(opts_df):
        t_all, S_all = [], []
        t_step = 1
        if name == 4.0 or name == 0.1:
            continue

        for idx, row in group.iterrows():
            sim_dir = row.simulation

            results = batch_process(
                sim_dir,
                method,
                alpha,
                max_workers,
                partcle_label_exclude,
                threshold=0.75,
            )
            x = [float(r[0]) for r in results]
            y = [r[1] for r in results]
            t_all.append(x)
            S_all.append(y)
            
        t_common = t_all[0]
        S_array = np.array([S for S in S_all])
        S_mean = np.mean(S_array, axis=0)
        S_std = np.std(S_array, axis=0)
        S_sem = S_std / np.sqrt(len(S_all))  # Standard error of the mean
        c = cmap(norm(gidx))
        C.append(c)
        X.append(t_common)
        Y.append(S_mean)
        Yerr.append(S_sem)
        L.append(f"{name}")

# %%
out_dir = "./af/"
os.makedirs(out_dir, exist_ok=True)

# Store numeric arrays directly
np.save(os.path.join(out_dir, "X.npy"), np.array(X, dtype=object), allow_pickle=True)
np.save(os.path.join(out_dir, "Y.npy"), np.array(Y, dtype=object), allow_pickle=True)
np.save(os.path.join(out_dir, "Yerr.npy"), np.array(Yerr, dtype=object), allow_pickle=True)
np.save(os.path.join(out_dir, "C.npy"), np.array(C, dtype=float))

# Optional CSV exports for easy inspection
pd.DataFrame({"X": [list(v) for v in X]}).to_csv(
    os.path.join(out_dir, "X.csv"), index=False
)
pd.DataFrame({"Y": [list(v) for v in Y]}).to_csv(
    os.path.join(out_dir, "Y.csv"), index=False
)
pd.DataFrame({"Yerr": [list(v) for v in Yerr]}).to_csv(
    os.path.join(out_dir, "Yerr.csv"), index=False
)
pd.DataFrame(C, columns=["r", "g", "b", "a"]).to_csv(
    os.path.join(out_dir, "C.csv"), index=False
)

# %%
# Load aggregated data from ./latest/
out_dir = "./af/"
csv_folder = "anisotropic_friction"
X = np.load(os.path.join(out_dir, "X.npy"), allow_pickle=True)
Y = np.load(os.path.join(out_dir, "Y.npy"), allow_pickle=True)
Yerr = np.load(os.path.join(out_dir, "Yerr.npy"), allow_pickle=True)
C = np.load(os.path.join(out_dir, "C.npy"))

# %%
odf = opts_dir_to_df(csv_folder)
# get time in units of doubling time
t_g = (
    odf.division_time_1[-1] / odf.time_step[-1]
)  # use the reference population division time as the generation time scaling

L = []  # better of doing this, for now manually input division lengths from data
fig, ax = plt.subplots()
for t_common, S_mean,S_err, name, c in zip(X, Y, Yerr, L, C):
    time = np.asarray([float(t) for t in t_common], dtype=float)
    y = np.asarray(S_mean, dtype=float)
    yerr = np.asarray(S_err, dtype=float)

    # remove NaN values at the end of the simulation where some sims live longer by random chance
    mask = np.isfinite(time) & np.isfinite(y) & np.isfinite(yerr)
    # alternatively, find the index there all have a valid data and use that

    # Ensure numeric arrays and filter invalid entries
    if np.any(mask):
        tc = np.asarray(time / t_g, dtype=float)
        yv = np.asarray(y[mask], dtype=float)
        ye = np.asarray(yerr[mask], dtype=float)

        tc = tc[mask]
        yv = yv[mask]
        ye = ye[mask]
        ax.plot(tc, yv, label=f"{name}", color=c)
        ax.fill_between(tc, yv - ye, yv + ye, alpha=0.3, color=c)

ax.set_xlabel("$t / T$", fontsize = 18)
ax.set_ylabel(r"$\phi_B$", fontsize=24)
ax.tick_params(axis="both", labelsize=16)
ax.legend(
    title=r"$\alpha_{AF}$",
    title_fontsize=16,
    fontsize=13,
    loc="upper left",
    ncol=2,
    framealpha=0.4,
)
plt.tight_layout()
fig
