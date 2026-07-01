import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.spatial import distance_matrix
import gc
import re


results_for_aspect_ratio_10 = {}

thickness = 0.1
trials = range(1, 6)


bins = np.linspace(0, 1, 50)  


def compute_radius(df):
    x, y = df[["x", "y"]].T.values
    center_x, center_y = np.mean(x), np.mean(y)
    distances_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    R = np.max(distances_to_center)
    return R


def compute_correlation(df, thickness, R, path):
    x, y, color2 = df[["x", "y", "color2"]].T.values

    
    match = re.search(r'(division_lengths|MAR):([\d.]+),([\d.]+)', path)
    if match:
        asp1 = round(float(match.group(2)) / 0.1, 0) if match.group(1) == 'division_lengths' else float(match.group(2))
        asp2 = round(float(match.group(3)) / 0.1, 0) if match.group(1) == 'division_lengths' else float(match.group(3))
    else:
        
        print(f"Regex did not match for path: {path}")
        return None, None, None

    
    if asp1 != 10 and asp2 != 10:
        return None, None, None

    
    if asp1 == 10.0:
        aspect_ratio_key = (asp1, asp2)
    elif asp2 == 10.0:
        aspect_ratio_key = (asp2, asp1)
    else:
        
        return None, None, None

    
    print(f"Processing aspect ratio key: {aspect_ratio_key}")

    
    positions = np.column_stack((x, y))

    
    distance_memmap = np.memmap('distance_matrix.dat', dtype='float32', mode='w+', shape=(len(x), len(x)))

    
    chunk_size = 1000  
    for i in range(0, len(x), chunk_size):
        for j in range(i, len(x), chunk_size):
            end_i = min(i + chunk_size, len(x))
            end_j = min(j + chunk_size, len(x))
            chunk_distances = distance_matrix(positions[i:end_i], positions[j:end_j])
            distance_memmap[i:end_i, j:end_j] = chunk_distances
            if i != j:
                distance_memmap[j:end_j, i:end_i] = chunk_distances  

    
    distance_memmap /= R

    bin_indices = np.digitize(distance_memmap, bins, right=True) - 1  

    
    same_type = color2[:, None] == color2  
    correlation_matrix = np.where(same_type, 1, -1)  

    
    num_bins = len(bins) - 1
    correlation_per_distance = np.zeros(num_bins)
    count_per_distance = np.zeros(num_bins)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):  
            bin_idx = bin_indices[i, j]
            if 0 <= bin_idx < num_bins:
                correlation_per_distance[bin_idx] += correlation_matrix[i, j]
                count_per_distance[bin_idx] += 1

    
    valid_bins = count_per_distance > 0
    avg_correlation = np.zeros(num_bins)
    avg_correlation[valid_bins] = correlation_per_distance[valid_bins] / count_per_distance[valid_bins]

    
    del distance_memmap
    gc.collect()

    
    

    return aspect_ratio_key, avg_correlation, bins[:-1]

def main():
    all_results = []
    aspect_folders = ['asp10', 'asp2', 'asp3', 'asp4', 'asp5', 'asp6', 'asp7', 'asp8', 'asp9', 'asp11', 'asp12', 'asp13', 'asp14', 'asp15']
    
    for trial in trials:
        for aspect_folder in aspect_folders:
            trial_path = f"/home/mratman1/activeMatterCommunities/workspace/simulations_eqdivtime/{aspect_folder}/run{trial}"
            
            if not os.path.exists(trial_path):
                print(f"Path does not exist: {trial_path}")
                continue

            subfolders = [f.path for f in os.scandir(trial_path) if f.is_dir()]
            
            
            print(f"Processing Trial {trial}, Aspect Folder '{aspect_folder}': {len(subfolders)} subfolders found.")

            for subfolder in subfolders:
                filenames = [f for f in os.listdir(subfolder) if f.endswith('.csv')]
                if not filenames:
                    print(f"No CSV files found in {subfolder}. Skipping.")
                    continue

                numeric_filenames = [f for f in filenames if f.split('.')[0].isdigit()]
                if not numeric_filenames:
                    print(f"No numeric CSV filenames in {subfolder}. Skipping.")
                    continue

                final_time = max(int(f.split('.')[0]) for f in numeric_filenames)

                
                csv_path = os.path.join(subfolder, f"{final_time}.csv")
                if not os.path.exists(csv_path):
                    print(f"CSV file does not exist: {csv_path}. Skipping.")
                    continue

                try:
                    chunk_iter = pd.read_csv(csv_path, chunksize=10000)
                except Exception as e:
                    print(f"Error reading CSV file {csv_path}: {e}. Skipping.")
                    continue

                for chunk in chunk_iter:
                    R = compute_radius(chunk)
                    aspect_ratio_key, avg_correlation, bins_values = compute_correlation(chunk, thickness, R, subfolder)

                    if aspect_ratio_key is None:
                        continue

                    if aspect_ratio_key not in results_for_aspect_ratio_10:
                        results_for_aspect_ratio_10[aspect_ratio_key] = {
                            "avg_correlations": [], "bins": bins_values
                        }

                    results_for_aspect_ratio_10[aspect_ratio_key]["avg_correlations"].append(avg_correlation)

                    
                    del chunk
                    gc.collect()

    
    averaged_results = {}

    for key, data in results_for_aspect_ratio_10.items():
        if not data["avg_correlations"]:
            continue  

        avg_correlations_all_trials = np.mean(data["avg_correlations"], axis=0)
        sem_correlations_all_trials = np.std(data["avg_correlations"], axis=0) / np.sqrt(len(data["avg_correlations"]))

        averaged_results[key] = {
            "avg_correlation": avg_correlations_all_trials,
            "sem_correlation": sem_correlations_all_trials,
            "bins": data["bins"]
        }

    
    filtered_aspects = {key: val for key, val in averaged_results.items() if 2 <= key[1] <= 15}

    
    for (asp1, asp2), data in filtered_aspects.items():
        df = pd.DataFrame({
            'bins': data['bins'],
            'avg_correlation': data['avg_correlation'],
            'sem_correlation': data['sem_correlation'],
            'asp1': asp1,
            'asp2': asp2
        })
        all_results.append(df)

    if all_results:
        
        results_df = pd.concat(all_results, ignore_index=True)

        
        results_df.to_csv("correlation_asp10_paper.csv", index=False)

        print("Results saved to correlation_asp10_paper.csv")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
