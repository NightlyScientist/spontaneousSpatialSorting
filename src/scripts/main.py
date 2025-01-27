import argparse
import os
import pathlib
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--cores", type=int, default=1)

# parser.add_argument("--cycles", type=int, default=10000)
parser.add_argument("--max_ptcls", type=int, default=500)
# parser.add_argument("--number_frames", type=int, default=100)
parser.add_argument("--frame_size", type=int, default=5000)
parser.add_argument("--system_size", type=float, default=50.0)

parser.add_argument("--time_step", type=float, default=0.0005)
parser.add_argument("--thickness", type=float, default=0.1)
parser.add_argument("--tracers", type=int, default=0)

parser.add_argument("--diffusion_constant", type=float, default=0.0)
parser.add_argument("--force_constant", type=float, default=100.0)

parser.add_argument(
    "--initial_type",
    type=str,
    default="uniform",
    choices=["one_cell", "uniform", "annulus"],
    help="choose initial population from: one_cell, uniform, annulus",
)

parser.add_argument("--fill_fraction", type=float, default=1.0)

parser.add_argument("--recycle_cells", nargs="?", const=1, type=int, default=0)

parser.add_argument("--initial_cells", type=int, default=10)
parser.add_argument("--rng_seed", type=int, default=1)

parser.add_argument("--recompile", nargs="?", const=1, type=int, default=0)
parser.add_argument("--slurm", nargs="?", const=1, type=int, default=0)
parser.add_argument("--save_path", default="workspace/simulations/")

parser.add_argument("--max_aspect_ratio", nargs="+", type=float, required=True)
parser.add_argument("--growth_rate", type=float, default=0.0005)
parser.add_argument("--equal_division_time", nargs="?", const=1, type=int, default=0)

args = parser.parse_args()


if args.rng_seed == 1:
    args.rng_seed = round(time.time())

if len(args.max_aspect_ratio) == 1:
    print("Using single species")
    args.max_aspect_ratio = np.array([args.max_aspect_ratio[0]] * 2)
else:
    print("using two species")
    args.max_aspect_ratio = sorted(np.array((args.max_aspect_ratio[0:2])))

args.division_length = np.array(args.max_aspect_ratio) * args.thickness

if args.equal_division_time == 1:
    scaled_growth_rate = (
        args.division_length[0] / args.division_length[1] * args.growth_rate
    )
    args.growth_rate = np.array([scaled_growth_rate, args.growth_rate])
else:
    args.growth_rate = np.array([args.growth_rate] * 2)


# caculate the number of needed cycles
def calculate_cycles(args):
    div_times = args.division_length / args.growth_rate / (2 * args.time_step)
    print(f"Division Times (cycles): {int(div_times[0])} and {int(div_times[1])}")


calculate_cycles(args)

args.cycles = 1

rename = {
    "rng_seed": "R",
    "max_ptcls": "MP",
    "frame_size": "FS",
    # "number_frames": "NF",
    "system_size": "S",
    "initial_type": "IT",
    "initial_cells": "IC",
    "diffusion_constant": "D",
    "force_constant": "K",
    "time_step": "dT",
    "thickness": "T",
    "max_aspect_ratio": "MAR",
    "growth_rate": "GR",
    "equal_division_time": "EDT",
}

opts = {}
to_csv = {}

for arg in vars(args):
    value = getattr(args, arg)

    if type(value) == list or type(value) == np.ndarray:
        opts[arg] = ",".join([str(v) for v in value])

        list_values = getattr(args, arg)
        for i in range(len(list_values)):
            to_csv[str(arg) + f"_{i+1}"] = str(list_values[i])
    else:
        to_csv[str(arg)] = str(getattr(args, arg))
        opts[arg] = str(value)

savePath = (
    args.save_path + "_".join([f"{v}:{opts[k]}" for k, v in rename.items()]) + "/"
)
print(f"\nOutput Directory:\n\t{savePath}\n")

opts["save_path"] = savePath
opts = " ".join([f"--{k} {v}" for k, v in opts.items()])
opts = opts.replace("True", "1").replace("False", "0")
print(f"Options:\n\t{opts}\n")

# create log folder and write options to file
logPath = savePath + "/logs/"
pathlib.Path(logPath).mkdir(parents=True, exist_ok=True)
with open(logPath + "log.txt", mode="w") as logFile:
    print(opts, file=logFile)

# create directory to save into
pathlib.Path(savePath).mkdir(parents=True, exist_ok=True)
files = filter(lambda t: "opts" not in t and t.endswith(".csv"), os.listdir(savePath))
if len(list(files)) > 0:
    print("Directory contains files. Review, and delete.")
    exit()

# create opt file to save input paramters
simInfoFile = f"{savePath}/opts.csv"
with open(simInfoFile, mode="w") as optsFile:
    headers, values = zip(*to_csv.items())
    print(",".join(headers), file=optsFile)
    print(",".join(values), file=optsFile)

# create logs folder
pathlib.Path(f"workspace/logs/slurm").mkdir(parents=True, exist_ok=True)

# check that .exe exists, else make it
if args.recompile == 1 or not os.path.exists("src/base/build/main.exe"):
    pathlib.Path(f"src/base/build").mkdir(parents=True, exist_ok=True)
    cmd_execute_make = """
        make -C src/base clean
        make -C src/base
    """
    os.system(cmd_execute_make)

if args.slurm == 1:
    cmd = "sbatch src/scripts/slurm_main.sh " + opts
    os.system(cmd)
else:
    cmd = "bash src/scripts/slurm_main.sh " + opts
    os.system(cmd)
