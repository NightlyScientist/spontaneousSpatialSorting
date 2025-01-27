import numpy as np
from tools.dataModels.measurements import Measurements
from tools.dataAPI.datamodel import DataModel
import alphashape
from scipy.spatial import KDTree
from numba import jit


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
    # return np.array([alleleMapping[a] for a in allele])
    return alleleMapping


def fetch_trajectories(dataPath, alpha=0.2):
    times = DataModel.extract_times(dataPath)[1]
    boundary = fetch_boundary_cells(dataPath, times, alpha=alpha)
    history = create_intial_snapshot(dataPath, times, boundary)
    fetch_history(dataPath, times, history)
    alleleMapping = map_allele_to_population(dataPath, times)
    return conform(history, alleleMapping)


def conform(history, alleleMapping):
    trajectories = []
    keys = sorted(history.keys(), reverse=True)
    for ith_cell in range(len(history[keys[0]])):
        x, y, ex, ey, r_order, allele, l, t, d_f = (
            np.empty(len(keys)),
            np.empty(len(keys)),
            np.empty(len(keys)),
            np.empty(len(keys)),
            np.empty(len(keys)),
            np.empty(len(keys)),
            np.empty(len(keys)),
            np.empty(len(keys)),
            np.empty(len(keys)),
        )
        for k in keys:
            x[k] = history[k][ith_cell][1]
            y[k] = history[k][ith_cell][2]
            ex[k] = history[k][ith_cell][3]
            ey[k] = history[k][ith_cell][4]
            r_order[k] = history[k][ith_cell][9]
            allele[k] = history[k][ith_cell][6]
            l[k] = history[k][ith_cell][5]
            t[k] = history[k][ith_cell][10]
            d_f[k] = history[k][ith_cell][11]

        population_label = alleleMapping[allele[0]]
        trajectories.append(
            Trajectory(x, y, ex, ey, l, allele, r_order, t, d_f, population_label)
        )
    return trajectories


def santize_history(history):
    # check for duplicate trejectories
    pass


def fetch_data(dataPath, times, index=None):
    indx = np.min([len(times) - 1, index]) if index is not None else len(times) - 1
    return Measurements(dataPath=dataPath, time=times[indx]), indx


def fetch_boundary_cells(dataPath, times, alpha=0.2):
    ref_data, indx = fetch_data(dataPath, times)

    # construct alpha shape
    points = np.column_stack((ref_data.x, ref_data.y))
    alpha_shape = alphashape.alphashape(points, alpha=0.2)

    # Extract boundary points from alpha shape
    coords = np.array(alpha_shape.exterior.coords.xy)

    # Build KDTree from data.x and data.y
    kdtree = KDTree(points)

    # Find points closest to coords within a radius of 0.5
    radius = 0.5
    closest_points = kdtree.query_ball_point(coords.T, radius)

    # Print the indices of the closest points
    boundary_points = []
    for nbors in closest_points:
        for i in nbors:
            boundary_points.append(i)
    boundary_points = np.sort(np.unique(boundary_points))
    return boundary_points


def create_intial_snapshot(dataPath, times, boundary_points):
    ref_data, indx = fetch_data(dataPath, times)

    # not optimal, but calculate measures here
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
        for i in boundary_points
    ]

    boundary_coords = fetch_boundary_coords(dataPath, times, indx)
    x = np.array([cell[1] for cell in cell_data])
    y = np.array([cell[2] for cell in cell_data])
    distances_front = perpendicular_distance(boundary_coords, x, y)
    for i in range(len(cell_data)):
        cell_data[i].append(distances_front[i])

    time_snapshots[indx] = cell_data
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
    alpha_shape = alphashape.alphashape(points, alpha=0.2)
    coords = np.array(alpha_shape.exterior.coords.xy)
    return coords


def perpendicular_distance(coords, x, y):
    #! should be replaced with a perpdendicular distance algorithm
    distances = np.zeros(len(x), dtype=np.float64)
    kdtree = KDTree(np.column_stack((coords[0], coords[1])))
    distances, _ = kdtree.query(np.column_stack((x, y)))
    return distances


# def get_line_intersection(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
#    s10_x = p1_x - p0_x
#    s10_y = p1_y - p0_y
#    s32_x = p3_x - p2_x
#    s32_y = p3_y - p2_y

#    denom = s10_x * s32_y - s32_x * s10_y
#    if denom == 0:
#        return -1, -1
#        # Collinear

#    denomPositive = denom > 0

#    s02_x = p0_x - p2_x
#    s02_y = p0_y - p2_y
#    s_numer = s10_x * s02_y - s10_y * s02_x
#    if (s_numer < 0) == denomPositive:
#        return -1, -1
#        # No collision

#    t_numer = s32_x * s02_y - s32_y * s02_x
#    if (t_numer < 0) == denomPositive:
#        return -1, -1
#        # No collision

#    if ((s_numer > denom) == denomPositive) or ((t_numer > denom) == denomPositive):
#        return -1, -1
#        # No collision

#    # Collision detected
#    t = t_numer / denom
#    i_x = p0_x + (t * s10_x)
#    i_y = p0_y + (t * s10_y)
#    return i_x, i_y


def fetch_history(dataPath, times, time_snapshots):
    start_index = np.max(list(time_snapshots.keys()))
    n_cells = len(time_snapshots[start_index])

    while start_index > 0:
        cell_data = [None for _ in range(n_cells)]

        # get most recent time snapshot
        current = time_snapshots[start_index]
        current_data, _ = fetch_data(dataPath, times, start_index)

        # get previous time snapshot
        new_data, _ = fetch_data(dataPath, times, start_index - 1)
        radial = new_data.radial_alignment()

        for ith_cell in range(n_cells):
            index, x, y, *_ = current[ith_cell]

            # find ancestor in previous time snapshot
            if index >= new_data.l.size:
                # scan list and find entry with matching ancestor
                ancestor_candidates = find_ancestor(
                    current_data.color,
                    current_data.ancestors,
                    current_data.ancestors[index],
                    current_data.color[index],
                    current_data.splits[index],
                    index,
                )

                if len(ancestor_candidates) == 0:
                    print("error finding ancestor.")
                    return current_data, new_data, index
                else:
                    # ensure that ancestor index is within the size of the new data
                    ptr = 0
                    ancestor_index = ancestor_candidates[ptr]
                    while (
                        ancestor_index >= new_data.l.size
                        and len(ancestor_candidates) >= ptr
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
        x = np.array([cell[1] for cell in cell_data])
        y = np.array([cell[2] for cell in cell_data])
        distances_front = perpendicular_distance(boundary_coords, x, y)
        for i in range(len(cell_data)):
            cell_data[i].append(distances_front[i])

        time_snapshots[start_index] = cell_data


def parse_ancestors(ancestors: str):
    tmp = np.array(ancestors.split(":"), dtype=int)
    return _parse_ancestors(tmp)


@jit(nopython=True)
def _parse_ancestors(tmp):
    nm = 0
    while tmp[nm] != 0:
        nm += 1
    return tmp[0:nm]


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


def is_candiate(ancestors, _ancestors, splits):
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


# @jit(nopython=True)
def _find_ancestor(ids, barcodes, target_barcode, target_id, splits, target_index):
    candidates = []

    last_index = target_index - 1
    for j in range(last_index, -1, -1):
        _id = ids[j]
        _ancestors = parse_ancestors(barcodes[j])

        if _id != target_id:
            continue

        if is_candiate(target_barcode, _ancestors, splits):
            candidates.append(j)
    return candidates
