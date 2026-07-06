import argparse
import os
import sys
import pathlib
import time
import numpy as np
from pathlib import Path
from copy import deepcopy


def get_active_branch_name():
    branch = os.popen("git rev-parse --abbrev-ref HEAD").read().strip()
    ssha = os.popen("git rev-parse --short HEAD").read().strip()
    return branch, ssha


def saveLogs(opts):
    c_opts = deepcopy(opts)
    c_opts.pop("save_path", None)

    branch, ssha = get_active_branch_name()
    options = f"git branch: {branch}\n"
    options += f"git commit hash: {ssha}\n"
    options += "\n".join("  + {}: {}".format(k, v) for k, v in c_opts.items())
    options += "\n"
    options += "  + command: python " + f"{' '.join(sys.argv)}\n"
    options += "  + command: bash " + f"{convert_opts_str(c_opts)}\n"
    options += f"start time: {time.ctime()}\n"
    options += "\n\n"
    print(options)

    savepath = opts["save_path"]
    pathlib.Path("workspace/logs/slurm").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{savepath}/logs").mkdir(parents=True, exist_ok=True)
    simInfoFile = f"{savepath}/logs/summary_simulation_info.log"
    with open(simInfoFile, mode="a") as historyFile:
        historyFile.write(options)


def settings(input_args, nth_experiment):
    opts = {}
    to_csv = {}

    _options = deepcopy(vars(input_args))
    _options["rng_seed"] = nth_experiment + input_args.rng_seed
    args = argparse.Namespace(**_options)

    for arg in _options:
        value = getattr(args, arg)

        if isinstance(value, (list, np.ndarray)):
            opts[arg] = ",".join([str(v) for v in value])

            list_values = getattr(args, arg)
            for i in range(len(list_values)):
                to_csv[str(arg) + f"_{i + 1}"] = str(list_values[i])

        else:
            to_csv[str(arg)] = str(value)
            opts[arg] = str(value)
    to_csv.pop("save_path", None)
    return opts, to_csv


def savePathCheck(savePath, rewrite=False):
    pathlib.Path(savePath).mkdir(parents=True, exist_ok=True)
    files = filter(
        lambda t: "opts" not in t and t.endswith(".csv"), os.listdir(savePath)
    )
    if len(list(files)) > 0:
        if not rewrite:
            print("Directory contains files. Review, and delete.")
            print(f"Directory: {savePath}\n")
            exit()
        for file_name in files:
            file_path = os.path.join(savePath, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)


def compile_exe(need_to_compile):
    if not need_to_compile and os.path.exists("src/base/build/main.exe"):
        return False

    cmd_execute_make = """
        make -C src/base clean
        make -C src/base
    """
    os.system(cmd_execute_make)
    return False


def convert_opts_str(opts):
    opts_str = " ".join([f"--{k} {v}" for k, v in opts.items()])
    opts_str = opts_str.replace("True", "1").replace("False", "0")
    return opts_str


def generatateOptsFile(opts, to_csv):
    # create opt file to save input paramters
    with open(f"{opts['save_path']}/opts.csv", mode="w") as optsFile:
        headers, values = zip(*to_csv.items())
        print(",".join(headers), file=optsFile)
        print(",".join(values), file=optsFile)


def capsule_area(aspect_ratio, thickness):
    """Area of a 2D capsule given aspect ratio and thickness."""
    return (aspect_ratio + np.pi / 4) * thickness**2


def scale_to_equal_area(a1, d1, a2, d2):
    """
    Returns the scaling factor s such that scaling capsule 1 (a1, d1)
    by s gives it the same area as capsule 2 (a2, d2), preserving aspect ratio.
    """
    area2 = capsule_area(a2, d2)
    area1 = capsule_area(a1, d1)
    s = np.sqrt(area2 / area1)
    return s


parser = argparse.ArgumentParser()
parser.add_argument("--rewrite", nargs="?", const=1, type=int, default=0)
parser.add_argument("--cores", type=int, default=1)
parser.add_argument("--max_ptcls", type=int, default=500)
parser.add_argument("--frame_size", type=int, default=5000)
parser.add_argument("--system_size", type=float, default=50.0)
parser.add_argument("--time_step", type=float, default=0.0005)
parser.add_argument("--tracers", type=int, default=0)
parser.add_argument("--growth_rate", type=float, default=0.0005)
parser.add_argument("--diffusion_constant", type=float, default=0.0)
parser.add_argument("--force_constant", type=float, default=100.0)
parser.add_argument("--anisotropic_friction", type=float, default=0.0)
parser.add_argument("--number_experiments", type=int, default=1)
parser.add_argument("--annulus_start_delay", type=int, default=0)
parser.add_argument("--initial_cells", type=int, default=10)
parser.add_argument("--rng_seed", type=int, default=0)
parser.add_argument("--recompile", nargs="?", const=1, type=int, default=0)
parser.add_argument("--slurm", nargs="?", const=1, type=int, default=0)
parser.add_argument("--save_path", default="workspace/simulations/")
parser.add_argument("--max_aspect_ratio", nargs="+", type=float, required=True)
parser.add_argument("--thickness", nargs="+", type=float, required=True)
parser.add_argument("--equal_division_time", nargs="?", const=1, type=int, default=0)
parser.add_argument("--no_sort", nargs="?", const=1, type=int, default=0)
parser.add_argument("--debug", nargs="?", const=1, type=int, default=0)
parser.add_argument("--equal_area", nargs="?", const=1, type=int, default=0)
parser.add_argument("--carrying_capacity", type=float, default=20.0)
parser.add_argument("--rate_adjust", type=float, default=0.005)
parser.add_argument(
    "--initial_type",
    type=str,
    default="uniform",
    choices=["one_cell", "uniform", "annulus", "passive_ring"],
    help="choose initial population from: one_cell, uniform, annulus",
)
args = parser.parse_args()

if args.rng_seed == 0:
    args.rng_seed = round(time.time())

# ensure the each  argument with optional two values has two values as copies
for arg in ["thickness", "max_aspect_ratio", "growth_rate"]:
    if arg in args:
        if isinstance(getattr(args, arg), (int, float)):
            setattr(args, arg, np.array([getattr(args, arg)] * 2))
        else:
            setattr(args, arg, np.array(getattr(args, arg)))
            if len(getattr(args, arg)) != 2:
                setattr(args, arg, np.array([getattr(args, arg)[0]] * 2))

# sort array based on max length
args.division_length = args.max_aspect_ratio * args.thickness
args.division_time = args.division_length / args.growth_rate
args.areas = capsule_area(args.max_aspect_ratio, args.thickness)

if args.equal_area == 1:
    # modify division length and thickness to maintain the aspect ratio and equal area
    cell_areas = capsule_area(args.max_aspect_ratio, args.thickness)
    sorter = np.argsort(cell_areas)
    a1, a2 = args.max_aspect_ratio[sorter]
    d1, d2 = args.thickness[sorter]
    s = scale_to_equal_area(a1, d1, a2, d2)
    new_d1 = s * d1
    new_L1 = a1 * new_d1

    # Update only the smaller capsule
    args.max_aspect_ratio[sorter[0]] = a1  # unchanged
    args.thickness[sorter[0]] = new_d1
    args.division_length[sorter[0]] = new_L1

    # Keep the larger capsule unchanged
    args.max_aspect_ratio[sorter[1]] = a2  # unchanged
    args.thickness[sorter[1]] = d2  # unchanged
    args.division_length[sorter[1]] = a2 * d2
    args.areas = capsule_area(args.max_aspect_ratio, args.thickness)
    args.division_time = args.division_length / args.growth_rate

if args.equal_division_time == 1:
    d_time1 = args.division_time[0]
    d_time2 = args.division_time[1]
    if d_time1 > d_time2:
        args.division_time[1] = d_time1
        args.growth_rate[1] = args.division_length[1] / d_time1
    else:
        args.division_time[0] = d_time2
        args.growth_rate[0] = args.division_length[0] / d_time2

sortqueue = [
    "thickness",
    "division_length",
    "division_time",
    "max_aspect_ratio",
    "growth_rate",
    "areas",
]
if args.no_sort != 1:
    idx_sort = np.argsort(args.division_time)
    for arg in sortqueue:
        old = getattr(args, arg)
        setattr(args, arg, np.array(old)[idx_sort])

rename = {
    "rng_seed": "R",
    "max_ptcls": "MP",
    "frame_size": "FS",
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
    "equal_area": "EA",
    "annulus_start_delay": "ASD",
    "anisotropic_friction": "AF",
    "carrying_capacity": "CC",
    "rate_adjust": "RA",
}

for arg in ["equal_area", "equal_division_time"]:
    if getattr(args, arg) == 1:
        rename.pop(arg, None)

if "passitve_ring" not in args.initial_type:
    rename.pop("annulus_start_delay")


def main(args, rename, need_to_compile):
    compile_flag = need_to_compile

    for nth_experiment in range(args.number_experiments):
        opts, to_csv = settings(args, nth_experiment)

        save_path = opts["save_path"]
        for k, v in rename.items():
            if k in opts:
                lval = opts[k]
                save_path += (
                    f"{v}:{lval:.2e}_" if isinstance(lval, float) else f"{v}:{lval}_"
                )

        opts["save_path"] = save_path.rstrip("_") + "/"
        print(f"\nOutput Directory:\n\t{opts['save_path']}\n")

        saveLogs(opts)
        savePathCheck(opts["save_path"], args.rewrite)
        generatateOptsFile(opts, to_csv)
        compile_flag = compile_exe(compile_flag)

        platform = (
            "sbatch" if args.slurm == 1 or args.number_experiments > 1 else "bash"
        )
        cmd = f"{platform} src/scripts/slurm_main.sh {convert_opts_str(opts)}"
        print(f"Options:\n\t{convert_opts_str(opts)}\n")
        os.system(cmd)


main(args, rename, args.recompile)
