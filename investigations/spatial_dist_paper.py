import numpy as np
import pandas as pd
import os
import alphashape
from scipy.spatial import KDTree
import re

calculation_alpha_parameter = 1.3


def extract_aspect_ratios_from_path(path):
    match = re.search(r'(division_lengths|MAR):([\d.]+),([\d.]+)', path)
    if match:
        asp1 = round(float(match.group(2)) / 0.1, 0) if match.group(1) == 'division_lengths' else float(match.group(2))
        asp2 = round(float(match.group(3)) / 0.1, 0) if match.group(1) == 'division_lengths' else float(match.group(3))
        return asp1, asp2
    raise ValueError("Neither 'division_lengths' nor 'MAR' found in path")

def generate_points_along_rod(xi, yi, length, orientation, num_points=5):
    """Generate fewer points along a rod to reduce computational cost."""
    dx = (length / 2) * np.cos(orientation)
    dy = (length / 2) * np.sin(orientation)
    points = np.linspace(0, 1, num_points)
    return np.column_stack([(xi - dx + 2 * dx * t, yi - dy + 2 * dy * t) for t in points]).T  

def calculate_periphery(x, y, lengths, orientations, alpha):
    """Calculate the concave hull of the colony using the rod positions."""
    rod_endpoints = np.empty((2 * len(x), 2))  
    for i, (xi, yi, length, orientation) in enumerate(zip(x, y, lengths, orientations)):
        dx = (length / 2) * np.cos(orientation)
        dy = (length / 2) * np.sin(orientation)
        rod_endpoints[2 * i] = (xi - dx, yi - dy)
        rod_endpoints[2 * i + 1] = (xi + dx, yi + dy)

    
    if rod_endpoints.shape[1] != 2:
        rod_endpoints = rod_endpoints.T  

    
    hull = alphashape.alphashape(rod_endpoints, alpha)
    return hull

def count_periphery_cells(x, y, lengths, orientations, color2, hull, com_x, com_y, distance_threshold):
    """Count how many cells of each aspect ratio are on the periphery."""
    hull_points = np.array(hull.exterior.xy).T
    kdtree = KDTree(hull_points)
    periphery_count_asp1, periphery_count_asp2 = 0, 0

    for xi, yi, length, orientation, col2 in zip(x, y, lengths, orientations, color2):
        rod_points = generate_points_along_rod(xi, yi, length, orientation)
        for point in rod_points:
            distance_to_hull, _ = kdtree.query(point)
            if distance_to_hull <= distance_threshold:
                if col2 == 0:
                    periphery_count_asp1 += 1
                else:
                    periphery_count_asp2 += 1
                break  

    return periphery_count_asp1, periphery_count_asp2


results_per_aspect_ratio = {}
thickness = 0.1
aspect_folders = ['asp10', 'asp2', 'asp3', 'asp4', 'asp5', 'asp6', 'asp7', 'asp8', 'asp9', 'asp11', 'asp12', 'asp13', 'asp14', 'asp15']

for trial in range(1, 6):
    for aspect_folder in aspect_folders:
        trial_path = f"/home/mratman1/activeMatterCommunities/workspace/simulations_eqdivtime/{aspect_folder}/run{trial}"
        
        if not os.path.exists(trial_path):
            print(f"Path does not exist: {trial_path}")
            continue
        
        subfolders = [f.path for f in os.scandir(trial_path) if f.is_dir()]

        for subfolder in subfolders:
            asp1, asp2 = extract_aspect_ratios_from_path(subfolder)
            if asp1 == 10.0 or asp2 == 10.0:

                csv_files = sorted([f for f in os.listdir(subfolder) if f.endswith('.csv') and f.strip('.csv').isdigit()],
                                key=lambda f: int(f.strip('.csv')))
                if not csv_files:
                    continue

                output1_temp, output2_temp = [], []
                for time_file in csv_files:
                    time = int(time_file.strip('.csv'))
                    df = pd.read_csv(f"{subfolder}/{time_file}")
                    x, y, ex, ey, lengths, color2 = df[["x", "y", "ex", "ey", "l", "color2"]].T.values
                    orientations = np.arctan2(ey, ex)

                    
                    com_x, com_y = np.mean(x), np.mean(y)
                    distances = np.sqrt((x - com_x) ** 2 + (y - com_y) ** 2)
                    radius = np.max(distances)

                    
                    hull = calculate_periphery(x, y, lengths, orientations, alpha=calculation_alpha_parameter)
                    distance_threshold = 0.3

                    
                    periphery_count_asp1, periphery_count_asp2 = count_periphery_cells(
                        x, y, lengths, orientations, color2, hull, com_x, com_y, distance_threshold
                    )

                    
                    total_cells = periphery_count_asp1 + periphery_count_asp2
                    periphery_fraction_asp1 = periphery_count_asp1 / total_cells
                    periphery_fraction_asp2 = periphery_count_asp2 / total_cells

                    output1_temp.append(periphery_fraction_asp1)
                    output2_temp.append(periphery_fraction_asp2)

                
                if asp1 == 10.0:
                    aspect_ratio_key = (asp1, asp2)
                    if aspect_ratio_key not in results_per_aspect_ratio:
                        results_per_aspect_ratio[aspect_ratio_key] = {
                            "output1": [], "output2": [], "times": []
                        }
                    results_per_aspect_ratio[aspect_ratio_key]["output1"].append(output1_temp)
                    results_per_aspect_ratio[aspect_ratio_key]["output2"].append(output2_temp)
                    results_per_aspect_ratio[aspect_ratio_key]["times"].append([int(f.strip('.csv')) for f in csv_files])
                elif asp2 == 10.0:
                    aspect_ratio_key = (asp1, asp2)
                    if aspect_ratio_key not in results_per_aspect_ratio:
                        results_per_aspect_ratio[aspect_ratio_key] = {
                            "output1": [], "output2": [], "times": []
                        }
                    results_per_aspect_ratio[aspect_ratio_key]["output1"].append(output2_temp)
                    results_per_aspect_ratio[aspect_ratio_key]["output2"].append(output1_temp)
                    results_per_aspect_ratio[aspect_ratio_key]["times"].append([int(f.strip('.csv')) for f in csv_files])

            else:
                continue





averaged_results = {}
for key, data in results_per_aspect_ratio.items():
    all_times = sorted(set(time for trial_times in data["times"] for time in trial_times))
    avg_output1, sem_output1 = [], []

    for time in all_times:
        trial_outputs = [data["output1"][trial_idx][data["times"][trial_idx].index(time)]
                         for trial_idx, trial_times in enumerate(data["times"]) if time in trial_times]
        avg_output1.append(np.mean(trial_outputs))
        sem_output1.append(np.std(trial_outputs) / np.sqrt(len(trial_outputs)))

    averaged_results[key] = {"avg_output1": avg_output1, "sem_output1": sem_output1, "times": all_times}


all_results = []
for (asp1, asp2), data in averaged_results.items():
    for i, time in enumerate(data["times"]):
        all_results.append({
            "Aspect_Ratio_1": asp1,
            "Aspect_Ratio_2": asp2,
            "Time": time / 5e5,  
            "Avg_Fraction_Asp1": data["avg_output1"][i],
            "SEM_Avg_Fraction_Asp1": data["sem_output1"][i]
        })

df_results = pd.DataFrame(all_results)
df_results.to_csv("all_10_periphery_fractions_3d.csv", index=False)


# sorted_aspects = dict(sorted(averaged_results.items(), key=lambda x: x[0]))
# cmap = sns.diverging_palette(250, 15, s=90, l=50, n=9, center="dark", as_cmap=True)
# norm = mcolors.TwoSlopeNorm(vmin=2, vcenter=10, vmax=15)

# fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')

# for (asp1, asp2), data in sorted_aspects.items():
#     times = np.array(data["times"]) / 5e5
#     avg_output1, sem_output1 = data["avg_output1"], data["sem_output1"]
#     color = 'black' if (asp1, asp2) == (10, 10) else cmap(norm(asp2))
#     linewidth = 3 if (asp1, asp2) == (10, 10) else 1

#     ax.fill_between(times, np.array(avg_output1) - np.array(sem_output1), np.array(avg_output1) + np.array(sem_output1),
#                     color=color, alpha=0.3)
#     ax.plot(times, avg_output1, '-', color=color, linewidth=linewidth)


# ax.axhline(1, color='black', linestyle='--', linewidth=1.2)
# ax.axhline(0.5, color='lime', linestyle='--', linewidth=1.2)


# ax.set_xlabel(r"Time ($t/T$)", fontsize=16)
# ax.set_ylabel(r"Fraction of $a=10$ Cells at Colony Periphery", fontsize=16)
# ax.tick_params(axis='both', labelsize=16)
# ax.set_ylim(0.4, 1.0)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
# cbar.set_label('Population B Aspect Ratio', fontsize=18)
# cbar.set_ticks(range(2, 16))
# cbar.ax.tick_params(labelsize=16)

# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig("SPATIAL_DIST_OCT1.png")
# plt.show()
