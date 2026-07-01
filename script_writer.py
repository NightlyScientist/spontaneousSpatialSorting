def write_script(initial_asp):

  division_length = round(initial_asp/10, 3)
  growth_rate = round(0.000005 * initial_asp, 7) 
  print(division_length)

  my_list = []
  for k in range(2, 16):
      calc = round(0.000005 * k, 7)  # Corrected the calculation
      i = calc
      j = 0.1*k
      initial = f"python src/scripts/main.py --cycles 2100000000 --frame_size 400000 --growth_rate {growth_rate} {i} --thickness 0.1 --system_size 150.0 --time_step 0.005 --division_lengths {division_length} {j} --cores 2 --time_step 0.005 --max_ptcls 25000 --slurm --initial_cells 50 --initial_type uniform --rng_seed 1712031202"
      my_list.append(initial)

  print(';'.join(my_list))