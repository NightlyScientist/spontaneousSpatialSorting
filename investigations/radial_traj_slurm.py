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

# Generate base paths for aspect ratios from 2 to 15
base_paths = [
    f"/home/mratman1/activeMatterCommunities/workspace/simulations_eqdivtime/asp{x}"
    for x in range(2, 16)
]

def extract_aspect_ratios_from_path(path):
    match = re.search(r'(division_lengths|MAR):([\d.]+),([\d.]+)', path)
    if match:
        asp1 = (
            round(float(match.group(2)) / 0.1, 0)
            if match.group(1) == 'division_lengths'
            else float(match.group(2))
        )
        asp2 = (
            round(float(match.group(3)) / 0.1, 0)
            if match.group(1) == 'division_lengths'
            else float(match.group(3))
        )
        
        # Ensure that aspect ratio 10 is treated as aspect ratio 1
        if asp2 == 10:
            asp2 = asp1
            asp1 = 10
        
        return asp1, asp2
    raise ValueError("Neither 'division_lengths' nor 'MAR' found in path")

def has_consecutive_radial_order(radial_order, threshold, consecutive_thresh):
    return np.any(
        np.convolve(radial_order > threshold, np.ones(consecutive_thresh, dtype=int), 'valid')
        == consecutive_thresh
    )

def calculate_probability(sample_path):
    """
    Calculates the probability for a given sample path.
    Returns a tuple containing the sample path, aspect ratios, and probability.
    If extraction fails or conditions aren't met, returns None for prob.
    """
    try:
        asp1, asp2 = extract_aspect_ratios_from_path(sample_path)
    except ValueError:
        return sample_path, None, None, None  # Include the sample_path

    # **Corrected Condition**: Proceed only if at least one aspect ratio is 10
    if asp1 != 10 and asp2 != 10:
        return sample_path, asp1, asp2, None  # Skip paths without aspect ratio 10

    # Fetch trajectories
    trajectories = ancestorTracker.fetch_trajectories(sample_path)

    total_periphery_cells = 0
    qualified_cells = 0

    for traj in trajectories:
        if traj.front_displacement[-1] < 0.5 and traj.front_displacement[0] >= 0.5:
            total_periphery_cells += 1
            if has_consecutive_radial_order(traj.radial, 0.8, radial_persistence_thresh):
                qualified_cells += 1

    # Calculate the probability
    probability = (
        qualified_cells / total_periphery_cells
        if total_periphery_cells > 0
        else 0
    )
    return sample_path, asp1, asp2, probability


def process_sample_path(sample_path):
    return calculate_probability(sample_path)

# Parallel computation using Pool
def parallel_calculation(sample_paths):
    with Pool(processes=4) as pool:  # Adjust '4' based on the number of CPU cores you want to use
        results = pool.map(process_sample_path, sample_paths)
    return results

# Gather sample paths from all base directories
all_sample_paths = []
for basePath in base_paths:
    df = parameterTable(basePath)  # Adjust this to correctly gather paths
    all_sample_paths.extend(df.basePath.values)

# Call the parallel computation function
results = parallel_calculation(all_sample_paths)

# Initialize data structure for probabilities and a list for failed paths
probabilities_dict = {}
failed_paths = []

# Process results
for result in results:
    sample_path, asp1, asp2, prob = result
    if asp1 is None or asp2 is None or prob is None:
        failed_paths.append(sample_path)
        continue  # Skip if aspect ratios could not be extracted or prob is None
    
    if asp1 not in probabilities_dict:
        probabilities_dict[asp1] = {}
    
    if asp2 not in probabilities_dict[asp1]:
        probabilities_dict[asp1][asp2] = []
    
    probabilities_dict[asp1][asp2].append(prob)

# Log failed paths
if failed_paths:
    print("The following sample paths could not be processed and were skipped:")
    for path in failed_paths:
        print(f"- {path}")

# Calculate mean and SEM probabilities for each aspect ratio of population 1 and 2
mean_probs = {}
sem_probs = {}

for asp1, asp2_dict in probabilities_dict.items():
    mean_probs[asp1] = {}
    sem_probs[asp1] = {}
    
    for asp2, probs in asp2_dict.items():
        if len(probs) == 0:
            mean = 0
            sem = 0
        else:
            mean = np.mean(probs)
            sem = (
                np.std(probs, ddof=1) / np.sqrt(len(probs))
                if len(probs) > 1
                else 0
            )
        mean_probs[asp1][asp2] = mean
        sem_probs[asp1][asp2] = sem

# Prepare data for plotting and saving to CSV
aspect_ratios_population_1 = sorted(mean_probs.keys())
aspect_ratios_population_2 = sorted(
    set(asp2 for asp2_dict in mean_probs.values() for asp2 in asp2_dict)
)

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
csv_file_path = "/home/mratman1/activeMatterCommunities/investigations/mean_bulk_ALL10.csv"
df_probabilities.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")

# Plotting with error bars as shaded regions
plt.figure(figsize=(12, 8))
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(aspect_ratios_population_1)))

fig, ax = plt.subplots(figsize=(12, 8))

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
    
    ax.plot(
        asp2_values, 
        prob_values, 
        label=f'Asp1 = {asp1}', 
        color=colors[i], 
        marker='o'
    )
    ax.fill_between(
        asp2_values,
        np.array(prob_values) - np.array(sem_values),
        np.array(prob_values) + np.array(sem_values),
        color=colors[i],
        alpha=0.3
    )

# Adding colorbar
norm = mpl.colors.Normalize(
    vmin=min(aspect_ratios_population_1), 
    vmax=max(aspect_ratios_population_1)
)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Aspect Ratio Population 1')
cbar.set_ticks(aspect_ratios_population_1)
cbar.ax.set_yticklabels([str(asp1) for asp1 in aspect_ratios_population_1])

plt.xlabel('Aspect Ratio Population 2', fontsize=14)
plt.ylabel('Probability of Cells at Periphery', fontsize=14)
plt.title(
    'Probability of Cells at Periphery vs Aspect Ratios with SEM', 
    fontsize=16
)
plt.legend(
    title='Population 1 Aspect Ratio', 
    bbox_to_anchor=(1.05, 1), 
    loc='upper left'
)
plt.grid(True)
plt.tight_layout()
plt.show()

print("Plotting completed.")
