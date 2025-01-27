import numpy as np
from tools.dataModels.measurements import Measurements
from tools.dataAPI.datamodel import DataModel
import alphashape
from scipy.spatial import KDTree
from numba import jit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class AncestorHistory:
    def __init__(self, dataPath):
        self.dataPath = dataPath


class Trajectory:
    def __init__(self, x, y, ex, ey, l, allele, radial, time, dist_front, population):
        self.x = x
        self.y = y
        self.ex = ex
        self.ey = ey
        self.l = l
        self.allele = allele[0]
        self.population = population
        self.front_displacement = dist_front
        self.radial = radial
        self.time = time


def map_allele_to_population(dataPath, times):
    data, _ = fetch_data(dataPath, times, 0)
    alleleMapping = dict()
    for i in range(data.l.size):
        alleleMapping[data.color[i]] = data.color2[i]
    return alleleMapping


def fetch_trajectories(dataPath, alpha=1.3):
    times = DataModel.extract_times(dataPath)[1]
    boundary = fetch_boundary_cells(dataPath, times, alpha=alpha)
    
    # **Identify Non-Boundary Cells**
    ref_data, indx = fetch_data(dataPath, times)
    all_cells = np.arange(len(ref_data.x))
    non_boundary = np.setdiff1d(all_cells, boundary)
    
    # Limit to 1000 random non-boundary cells
    if len(non_boundary) > 1000:
        non_boundary = np.random.choice(non_boundary, 1000, replace=False)
    
    logging.info(f"Total cells: {len(all_cells)}, Boundary cells: {len(boundary)}, Non-boundary cells: {len(non_boundary)}")
    
    # **Use Non-Boundary Cells for Initial Snapshot**
    history = create_initial_snapshot(dataPath, times, non_boundary)
    fetch_history(dataPath, times, history)
    alleleMapping = map_allele_to_population(dataPath, times)
    return conform(history, alleleMapping)


# The rest of the code remains the same.



def conform(history, alleleMapping):
    trajectories = []
    keys = sorted(history.keys(), reverse=True)
    logging.info(f"Processing {len(history[keys[0]])} trajectories.")
    
    for ith_cell in range(len(history[keys[0]])):
        # Initialize arrays
        x = np.empty(len(keys))
        y = np.empty(len(keys))
        ex = np.empty(len(keys))
        ey = np.empty(len(keys))
        r_order = np.empty(len(keys))
        allele = np.empty(len(keys), dtype=object)  # Use dtype=object for strings
        l = np.empty(len(keys))
        t = np.empty(len(keys))
        d_f = np.empty(len(keys))

        skip_cell = False  # Flag to skip if any snapshot is missing

        for k_idx, k in enumerate(keys):
            cell = history[k][ith_cell]
            if cell is None:
                logging.warning(f"Missing data for cell {ith_cell} at time {k}. Skipping this trajectory.")
                skip_cell = True
                break  # Skip this trajectory as it has missing data
            if len(cell) != 12:
                logging.warning(f"cell_data[{ith_cell}] at time {k} has {len(cell)} elements instead of 12. Skipping this trajectory.")
                skip_cell = True
                break
            # **Unpack with 12 Elements**
            index, x_val, y_val, _, _, _, _, _, _, _, _, _ = cell

            x[k_idx] = x_val
            y[k_idx] = y_val
            ex[k_idx] = cell[3]
            ey[k_idx] = cell[4]
            r_order[k_idx] = cell[9]
            allele[k_idx] = cell[6]
            l[k_idx] = cell[5]
            t[k_idx] = cell[10]
            d_f[k_idx] = cell[11]

        if skip_cell:
            continue  # Skip this trajectory due to missing data

        population_label = alleleMapping.get(allele[0], "Unknown")
        trajectories.append(
            Trajectory(x, y, ex, ey, l, allele, r_order, t, d_f, population_label)
        )
    
    logging.info(f"Generated {len(trajectories)} trajectories.")
    return trajectories


def santize_history(history):
    # Check for duplicate trajectories
    pass


def fetch_data(dataPath, times, index=None):
    indx = np.min([len(times) - 1, index]) if index is not None else len(times) - 1
    return Measurements(dataPath=dataPath, time=times[indx]), indx


def fetch_boundary_cells(dataPath, times, alpha=1.3):
    ref_data, indx = fetch_data(dataPath, times)

    # Construct alpha shape
    points = np.column_stack((ref_data.x, ref_data.y))
    alpha_shape = alphashape.alphashape(points, alpha=alpha)

    # Extract boundary points from alpha shape
    coords = np.array(alpha_shape.exterior.coords.xy)

    # Build KDTree from data.x and data.y
    kdtree = KDTree(points)

    # Find points closest to coords within a radius of 0.5
    radius = 0.5
    closest_points = kdtree.query_ball_point(coords.T, radius)

    # Collect unique boundary point indices
    boundary_points = []
    for nbors in closest_points:
        for i in nbors:
            boundary_points.append(i)
    boundary_points = np.sort(np.unique(boundary_points))
    logging.info(f"Identified {len(boundary_points)} boundary cells.")
    return boundary_points


def create_initial_snapshot(dataPath, times, cell_indices):
    ref_data, indx = fetch_data(dataPath, times)

    # Calculate radial alignment
    radial = ref_data.radial_alignment()

    time_snapshots = dict()
    cell_data = [
        [
            i,
            ref_data.x[i],
            ref_data.y[i],
            ref_data.ex[i],
            ref_data.ey[i],
            ref_data.l[i],
            ref_data.color[i],
            ref_data.ancestors[i],
            ref_data.splits[i],
            radial[i],
            ref_data.time,
        ]
        for i in cell_indices
    ]

    boundary_coords = fetch_boundary_coords(dataPath, times, indx)
    x = np.array([cell[1] for cell in cell_data])
    y = np.array([cell[2] for cell in cell_data])
    distances_front = perpendicular_distance(boundary_coords, x, y)
    for i in range(len(cell_data)):
        cell_data[i].append(distances_front[i])  # Now 12 elements
        if len(cell_data[i]) != 12:
            logging.warning(f"Initial snapshot cell_data[{i}] has {len(cell_data[i])} elements instead of 12.")
    
    time_snapshots[indx] = cell_data
    logging.info(f"Created initial snapshot with {len(cell_data)} cells.")
    return time_snapshots


def fetch_radii_vs_time(dataPath):
    times = DataModel.extract_times(dataPath)[1]
    radius_vs_time = np.empty(len(times), dtype=np.float64)
    for i in range(len(times)):
        current_data, _ = fetch_data(dataPath, times, i)
        com = np.array([np.mean(current_data.x), np.mean(current_data.y)])
        radius = np.hypot(current_data.x - com[0], current_data.y - com[1])
        r_max = np.max(radius)
        mask = np.where((0.9 * r_max < radius) & (radius <= r_max))
        radius_vs_time[i] = np.mean(radius[mask])
    return radius_vs_time


def fetch_boundary_coords(dataPath, times, index):
    current_data, _ = fetch_data(dataPath, times, index)
    points = np.column_stack((current_data.x, current_data.y))
    alpha_shape = alphashape.alphashape(points, alpha=1.3)
    coords = np.array(alpha_shape.exterior.coords.xy)
    return coords


def perpendicular_distance(coords, x, y):
    # Replace with a proper perpendicular distance algorithm if needed
    distances = np.zeros(len(x), dtype=np.float64)
    kdtree = KDTree(np.column_stack((coords[0], coords[1])))
    distances, _ = kdtree.query(np.column_stack((x, y)))
    return distances


def fetch_history(dataPath, times, time_snapshots):
    start_index = np.max(list(time_snapshots.keys()))
    n_cells = len(time_snapshots[start_index])

    while start_index > 0:
        cell_data = [None for _ in range(n_cells)]

        # Get most recent time snapshot
        current = time_snapshots[start_index]
        current_data, _ = fetch_data(dataPath, times, start_index)

        # Get previous time snapshot
        new_data, _ = fetch_data(dataPath, times, start_index - 1)
        radial = new_data.radial_alignment()

        for ith_cell in range(n_cells):
            current_cell = current[ith_cell]
            if current_cell is None:
                continue  # Skip if current cell data is None

            # **Unpack with 12 Elements**
            index, x, y, _, _, _, _, _, _, _, _, _ = current_cell

            # Find ancestor in previous time snapshot
            if index >= new_data.l.size:
                # Scan list and find entry with matching ancestor
                ancestor_candidates = find_ancestor(
                    current_data.color,
                    current_data.ancestors,
                    current_data.ancestors[index],
                    current_data.color[index],
                    current_data.splits[index],
                    index,
                )

                if len(ancestor_candidates) == 0:
                    logging.error(f"Error finding ancestor for cell index {ith_cell} at time index {start_index}. Skipping this cell.")
                    cell_data[ith_cell] = None  # Optionally, assign default values
                    continue  # Skip processing this cell
                else:
                    # Ensure that ancestor index is within the size of the new data
                    ptr = 0
                    ancestor_index = ancestor_candidates[ptr]
                    while (
                        ancestor_index >= new_data.l.size
                        and ptr < len(ancestor_candidates) - 1
                    ):
                        ptr += 1
                        ancestor_index = ancestor_candidates[ptr]
                    use_index = ancestor_index
            else:
                use_index = index

            if use_index is not None:
                cell_data[ith_cell] = [
                    use_index,
                    new_data.x[use_index],
                    new_data.y[use_index],
                    new_data.ex[use_index],
                    new_data.ey[use_index],
                    new_data.l[use_index],
                    new_data.color[use_index],
                    new_data.ancestors[use_index],
                    new_data.splits[use_index],
                    radial[use_index],
                    new_data.time,
                ]

        start_index -= 1

        boundary_coords = fetch_boundary_coords(dataPath, times, start_index)
        # Extract x and y only for non-None cells
        x_coords = [cell[1] for cell in cell_data if cell is not None]
        y_coords = [cell[2] for cell in cell_data if cell is not None]
        x = np.array(x_coords)
        y = np.array(y_coords)
        distances_front = perpendicular_distance(boundary_coords, x, y)
        distance_idx = 0  # To track the distance_front for non-None cells

        for i in range(len(cell_data)):
            if cell_data[i] is not None:
                cell_data[i].append(distances_front[distance_idx])  # Now 12 elements
                if len(cell_data[i]) != 12:
                    logging.warning(f"fetch_history: cell_data[{i}] has {len(cell_data[i])} elements after appending distance_front.")
                distance_idx += 1

        time_snapshots[start_index] = cell_data


def parse_ancestors(ancestors: str):
    tmp = np.array(ancestors.split(":"), dtype=int)
    return _parse_ancestors(tmp)


@jit(nopython=True)
def _parse_ancestors(tmp):
    nm = 0
    while nm < len(tmp) and tmp[nm] != 0:
        nm += 1
    return tmp[:nm]


@jit(nopython=True)
def compare_ancestors(ancestors, _ancestors):
    overlap = 0
    _range = len(ancestors) if len(ancestors) < len(_ancestors) else len(_ancestors)
    for j in range(_range):
        if _ancestors[j] == ancestors[j]:
            overlap += 1
        else:
            break
    return overlap


def is_candidate(ancestors, _ancestors, splits):
    overlap = compare_ancestors(ancestors, _ancestors)
    return overlap >= len(ancestors) - splits


def find_ancestor(ids, barcodes, target_barcode, target_id, splits, target_index):
    if isinstance(target_barcode, str):
        _target_barcode = parse_ancestors(target_barcode)
    else:
        _target_barcode = target_barcode
    return _find_ancestor(
        ids, barcodes, _target_barcode, target_id, splits, target_index
    )


def _find_ancestor(ids, barcodes, target_barcode, target_id, splits, target_index):
    candidates = []

    last_index = target_index - 1
    for j in range(last_index, -1, -1):
        _id = ids[j]
        _ancestors = parse_ancestors(barcodes[j])

        if _id != target_id:
            continue

        if is_candidate(target_barcode, _ancestors, splits):
            candidates.append(j)
    return candidates
