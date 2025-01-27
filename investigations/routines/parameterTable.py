from numba import jit
import os
import numpy as np
import pandas as pd


def extract_times(dataPath: str):
    opts = pd.read_csv(f"{dataPath}/opts.csv", header=0).to_dict(orient="index")[0]
    times = list(
        map(
            lambda t: int(t.split(".")[0]),
            filter(
                lambda t: "opts" not in t and t.endswith(".csv"),
                os.listdir(dataPath),
            ),
        )
    )
    times.sort()

    return opts, times


def parameterTable(basePath):
    legacy = {
        "division_length_1": "division_lengths_1",
        "division_length_2": "division_lengths_2",
        "growth_rate_1": "growth_rates_1",
        "growth_rate_2": "growth_rates_2",
    }

    additions = {
        "max_aspect_ratio_1": "division_lengths_1",
        "max_aspect_ratio_2": "division_lengths_2",
    }

    optDictList = []
    for root, dirs, _ in os.walk(basePath):
        for dir in dirs:
            if (
                dir.startswith("cycles:")
                or dir.startswith("simulationsR:")
                or dir.startswith("R:")
            ):
                subPath = os.path.join(root, dir)
                opts, _ = extract_times(subPath)

                opts["basePath"] = subPath
                # .convert keys to adapt to the legacy keys
                for k in legacy.keys():
                    if k in opts:
                        opts[legacy[k]] = opts.pop(k)

                # .inserting max_aspect ratio field
                for k in additions.keys():
                    if k not in opts:
                        opts[k] = round(
                            float(opts[additions[k]]) / float(opts["thickness"]), 2
                        )

                div_time_1 = opts["division_lengths_1"] / opts["growth_rates_1"]
                div_time_2 = opts["division_lengths_2"] / opts["growth_rates_2"]
                opts["division_time_1"] = div_time_1
                opts["division_time_2"] = div_time_2

                # .check fr equal division time
                if "equal_division_time" not in opts:
                    if div_time_1 == div_time_2:
                        opts["equal_division_time"] = 1
                    else:
                        opts["equal_division_time"] = 0
                optDictList.append(opts)
    return pd.DataFrame.from_dict(optDictList)
