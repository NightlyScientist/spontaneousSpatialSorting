import pandas as pd
import numpy as np
import os
from typing import Any, Type


class SimulationParamaeters:
    def __init__(self, dataPath: str):
        opts = pd.read_csv(f"{dataPath}/opts.csv", header=0).to_dict(orient="index")[0]

        # add legacy keys
        self.division_lengths_1 = opts.get("division_lengths_1", np.nan)
        self.division_lengths_2 = opts.get("division_lengths_2", np.nan)
        self.growth_rates_1 = opts.get("growth_rates_1", np.nan)
        self.growth_rates_2 = opts.get("growth_rates_2", np.nan)
        self.thickness = opts.get("thickness", np.nan)

        # current parameters
        self.thickness_1 = opts.get("thickness_1", self.thickness)
        self.thickness_1 = opts.get("thickness_1", self.thickness)
        self.time_step = opts.get("time_step", np.nan)
        self.division_length_1 = opts.get("division_length_1", self.division_lengths_1)
        self.division_length_2 = opts.get("division_length_2", self.division_lengths_2)
        self.diffusion_constant = opts.get("diffusion_constant", np.nan)
        self.force_constant = opts.get("force_constant", np.nan)
        self.initial_type = opts.get("initial_type", "default")
        self.growth_rate_1 = opts.get("growth_rate_1", self.growth_rates_1)
        self.growth_rate_2 = opts.get("growth_rate_2", self.growth_rates_2)
        self.rng_seed = opts.get("rng_seed", np.nan)
        self.initial_cells = opts.get("initial_cells", np.nan)
        self.system_size = opts.get("system_size", np.nan)


def getattr_(obj, attr_name, dtype: Type[Any] = int):
    """Get an attribute from an object, returning a default value if it doesn't exist."""
    data = getattr(obj, attr_name, None)
    if data is not None:
        data = np.array(data, dtype=dtype)
    else:
        data = np.zeros(0, dtype=dtype)
    return data


class DataModel(SimulationParamaeters):
    def __init__(self, dataPath: str, time: int):
        super().__init__(dataPath)

        df = pd.read_csv(f"{dataPath}/{time}.csv", header=0)
        self.x = getattr_(df, "x", float)
        self.y = getattr_(df, "y", float)
        self.ex = getattr_(df, "ex", float)
        self.ey = getattr_(df, "ey", float)
        self.l = getattr_(df, "l", float)
        self.color = getattr_(df, "color", int)
        self.color2 = getattr_(df, "color2", int)
        self.fx = getattr_(df, "fx", float)
        self.fy = getattr_(df, "fy", float)
        self.g = getattr_(df, "g", float)
        self.splits = getattr_(df, "splits", int)
        self.id_1 = getattr_(df, "id_1", int)
        self.id_2 = getattr_(df, "id_2", int)
        self.ancestors = getattr_(df, "ancestors", str)
        self.ancestor = getattr_(df, "ancestor", int)
        self.gadj = getattr_(df, "gadj", float)
        self.stale = getattr_(df, "stale", bool)
        self.cell_r = getattr_(df, "cell_r", float)

        # .set phi in range [0, 2pi)
        ey = self.ey if self.ey is not None else np.zeros_like(self.x)
        ex = self.ex if self.ex is not None else np.zeros_like(self.x)
        self.phi = np.arctan2(ey, ex) + np.pi

        # .add missing keys
        self.max_aspect_ratio_1 = self.division_length_1 / self.thickness
        self.max_aspect_ratio_2 = self.division_length_2 / self.thickness
        self.r = self.thickness / 2.0

        # .set local radius cutoff default value
        self.local_radius = (3 / 16) * (self.division_length_1 + self.division_length_2)
        # self.local_radius = (3 / 8) * (self.division_length_1 + self.division_length_2)

        # .some may have time as a field
        if "time" in df:
            self.time = df["time"].values[0]
        else:
            self.time = time * self.time_step

    def set_local_radius(self, radius: float):
        self.local_radius = radius

    @staticmethod
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
