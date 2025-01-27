import pandas as pd
import numpy as np
import tools.tracking.ancestors as ancestorTracker
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib as mpl
from routines.parameterTable import parameterTable
import re
from importlib import reload
from multiprocessing import Pool  # For parallel processing

# Threshold for consecutive radial_order > 0.8
radial_persistence_thresh = 10  # Adjust as needed
reload(ancestorTracker)

# Read CSV from path
basePath = "/home/mratman1/activeMatterCommunities/workspace/simulations_eqdivtime/asp10"
df = parameterTable(basePath)

def extract_aspect_ratios_from_path(path):
    match = re.search(r'(division_lengths|MAR):([\d.]+),([\d.]+)', path)
    if match:
        asp1 = round(float(match.group(2)) / 0.1, 0) if match.group(1) == 'division_lengths' else float(match.group(2))
        asp2 = round(float(match.group(3)) / 0.1, 0) if match.group(1) == 'division_lengths' else float(match.group(3))
        return asp1, asp2
    raise ValueError("Neither 'division_lengths' nor 'MAR' found in path")

def has_consecutive_radial_order(radial_order, threshold, consecutive_thresh):
    return np.any(np.convolve(radial_order > threshold, np.ones(consecutive_thresh, dtype=int), 'valid') == consecutive_thresh)

def calculate_probability(sample_path):
    # Extract aspect ratios from the file path
    try:
        asp1, asp2 = extract_aspect_ratios_from_path(sample_path)
    except ValueError:
        return None, None, None  # Skip this sample path if aspect ratios cannot be extracted

    # Fetch trajectories
    trajectories = ancestorTracker.fetch_trajectories(sample_path)
    
    total_periphery_cells = 0
    qualified_cells = 0

    for traj in trajectories:
        if traj.front_displacement[-1] < 0.5 and traj.front_displacement[0] >= 0.2:
            total_periphery_cells += 1
            if has_consecutive_radial_order(traj.radial, 0.8, radial_persistence_thresh):
                qualified_cells += 1

    # Calculate the probability
    probability = qualified_cells / total_periphery_cells if total_periphery_cells > 0 else 0
    return asp1, asp2, probability

def process_sample_path(sample_path):
    return calculate_probability(sample_path)

# Parallel computation using Pool
def parallel_calculation(sample_paths):
    with Pool(processes=4) as pool:  # Adjust '4' based on the number of CPU cores you want to use
        results = pool.map(process_sample_path, sample_paths)
    return results

# Call the parallel computation function
sample_paths = df.basePath.values
results = parallel_calculation(sample_paths)

# Initialize data structure for probabilities
probabilities_dict = {}

# Process results
for result in results:
    if result is None:
        continue  # Skip if aspect ratios could not be extracted
    
    asp1, asp2, prob = result
    
    if asp1 not in probabilities_dict:
        probabilities_dict[asp1] = {}
    
    if asp2 not in probabilities_dict[asp1]:
        probabilities_dict[asp1][asp2] = []
    
    probabilities_dict[asp1][asp2].append(prob)

# Calculate mean and SEM probabilities for each aspect ratio of population 1 and 2
mean_probs = {}
sem_probs = {}

for asp1, asp2_dict in probabilities_dict.items():
    mean_probs[asp1] = {}
    sem_probs[asp1] = {}
    
    for asp2, probs in asp2_dict.items():
        mean_probs[asp1][asp2] = np.mean(probs)
        sem_probs[asp1][asp2] = np.std(probs, ddof=1) / np.sqrt(len(probs)) if len(probs) > 1 else 0

# Prepare data for plotting and saving to CSV
aspect_ratios_population_1 = sorted(mean_probs.keys())
aspect_ratios_population_2 = sorted(set(asp2 for asp2_dict in mean_probs.values() for asp2 in asp2_dict))

# Save data to CSV
data_to_save = []
for asp1 in aspect_ratios_population_1:
    for asp2, prob in mean_probs[asp1].items():
        data_to_save.append({
            'Aspect_Ratio_Population_1': asp1, 
            'Aspect_Ratio_Population_2': asp2, 
            'Mean_Probability': prob,
            'SEM': sem_probs[asp1][asp2]
        })

df_probabilities = pd.DataFrame(data_to_save)
csv_file_path = "mean_radial_init_criterion_02_periph.csv"
df_probabilities.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")

# Plotting with error bars as shaded regions
plt.figure(figsize=(12, 8))
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(aspect_ratios_population_1)))

fig, ax = plt.subplots(figsize=(12, 8))
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(aspect_ratios_population_1)))

for i, asp1 in enumerate(aspect_ratios_population_1):
    asp2_values = []
    prob_values = []
    sem_values = []
    
    for asp2 in aspect_ratios_population_2:
        if asp2 in mean_probs[asp1]:
            asp2_values.append(asp2)
            prob_values.append(mean_probs[asp1][asp2])
            sem_values.append(sem_probs[asp1][asp2])
    
    if len(asp2_values) == 0:
        continue
    
    ax.plot(asp2_values, prob_values, label=f'Asp1 = {asp1}', color=colors[i], marker='o')
    ax.fill_between(
        asp2_values,
        np.array(prob_values) - np.array(sem_values),
        np.array(prob_values) + np.array(sem_values),
        color=colors[i],
        alpha=0.3
    )

# Adding colorbar
norm = mpl.colors.Normalize(vmin=min(aspect_ratios_population_1), vmax=max(aspect_ratios_population_1))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Aspect Ratio Population 1')
cbar.set_ticks(aspect_ratios_population_1)
cbar.ax.set_yticklabels([str(asp1) for asp1 in aspect_ratios_population_1])

plt.xlabel('Aspect Ratio Population 2', fontsize=14)
plt.ylabel('Probability of Cells at Periphery', fontsize=14)
plt.title('Probability of Cells at Periphery vs Aspect Ratios with SEM', fontsize=16)
plt.legend(title='Population 1 Aspect Ratio', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

print("Plotting completed.")
