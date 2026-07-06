#include "dataManager.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "SimulationContext.h"

SimulationParameters parseArgs(const std::vector<std::string> &args) {
  SimulationParameters config{};
  std::unordered_map<std::string, std::string> argMap;

  for (size_t i = 1; i < args.size(); i += 2) {
    if (i + 1 < args.size()) {
      argMap[args[i]] = args[i + 1];
    }
  }

  // Print key-value pairs to stdout
  std::cout << "Parsed command line arguments:" << std::endl;
  for (const auto &kv : argMap) {
    std::cout << kv.first << " == " << kv.second << std::endl;
  }

  config.cpu_cores = convert<int>(argMap, "--CPUs");
  config.maxPtcls = convert<int>(argMap, "--maxPtcls");
  config.work_dir = convert<std::string>(argMap, "--dir");
  config.rng_seed = convert<int>(argMap, "--seed");
  config.frameSize = convert<int>(argMap, "--frameSize");
  config.systemSize = convert<int>(argMap, "--systemSize");
  config.del_t = convert<double>(argMap, "--del_t");
  config.tracers = convert<int>(argMap, "--ntracers");
  config.diffusion_trans = convert<double>(argMap, "--diffusion_trans");
  config.f_0 = convert<double>(argMap, "--force");
  config.annulus_start_delay = convert<int>(argMap, "--annulus_start_delay");
  config.initial_cells = convert<int>(argMap, "--initialCells");
  config.ANISOTROPIC_FRICTION = convert<double>(argMap, "--anisotropic_friction");
  config.RateAdjust = convert<double>(argMap, "--rate_adjust");
  config.CarryingCapacity = convert<double>(argMap, "--carrying_capacity");
  config.division_lengths = splitAndConvert<double>(argMap, "--division_lengths");
  config.growth_rates = splitAndConvert<double>(argMap, "--growth_rates");
  config.thickness_arr = splitAndConvert<double>(argMap, "--thickness");
  config.initial_type = argMap["--initial_type"];
  return config;
}

void read_parameters(int argc, char *argv[], SimulationParameters &config) {
  std::vector<std::string> args(argv, argv + argc);
  config = parseArgs(args);

  // update force constant to be scaled by thickness
  double d0_mean = (config.thickness_arr[0] + config.thickness_arr[1]) / 2;
  config.f_0 *= std::sqrt(d0_mean);

  const auto del_t = config.del_t;
  config.diffusion_rot = config.diffusion_trans * 2.0;
  config.eff_diff_rot = sqrt(del_t * 2.0 * config.diffusion_rot);
  config.etdiff = sqrt(del_t * 2.0 * config.diffusion_trans);
}

void save_data_csv(int l_itime, double time, const SimulationContext &ctx) {
  std::string fname = ctx.work_dir + "/" + std::to_string(l_itime) + ".csv";
  CsvWriter csv(fname);

  // Write header row
  csv.write_header({"x", "y", "ex", "ey", "l", "ancestor", "color", "color2", "fx", "fy", "ancestors", "splits", "phi",
                    "g", "gadj", "stale", "time", "cell_r"});

  const auto &ptcl = ctx.get_particles();

  for (size_t i = 0; i < ptcl.size(); ++i) {
    const auto &p = ptcl[i];
    int color, color2;
    color = static_cast<int>(p.type);
    color2 = p.ancestor_rnd_group;

    std::string ancestors = "";
    for (size_t i = 0; i < p.ancestor_list.size(); ++i) {
      ancestors += std::to_string(p.ancestor_list[i]);
      if (i < p.ancestor_list.size() - 1) {
        ancestors += ":";
      }
    }

    std::string str = std::to_string(p.pos[0]) + "," + std::to_string(p.pos[1]) + "," + std::to_string(p.ori[0]) + "," +
                      std::to_string(p.ori[1]) + "," + std::to_string(p.length) + "," + std::to_string(p.ancestor) +
                      "," + std::to_string(color) + "," + std::to_string(color2) + "," +
                      std::to_string(p.forces[0] / p.length) + "," + std::to_string(p.forces[1] / p.length) + "," +
                      ancestors + "," + std::to_string(p.splits) + "," + std::to_string(p.phi) + "," +
                      std::to_string(p.growth_rate) + "," + std::to_string(p.growth_factor) + "," +
                      std::to_string(p.stale) + "," + std::to_string(time) + "," + std::to_string(p.thickness / 2);
    csv.write_string(str);
  }
}
