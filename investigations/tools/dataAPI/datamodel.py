import pandas as pd
import numpy as np
import os


class DataModel:
    def __init__(self, dataPath: str, time: int):
        opts = pd.read_csv(f"{dataPath}/opts.csv", header=0).to_dict(orient="index")[0]
        df = pd.read_csv(f"{dataPath}/{time}.csv", header=0)

        # .data fields from simuation
        keys = ["x", "y", "ex", "ey", "l", "color", "color2", "fx", "fy", "g", "splits", "id_1", "id_2", "ancestors", "ancestor"]
        for key in keys:
            setattr(self, key, df[key].values)


        # .set phi in range [0, 2pi)
        self.phi = np.arctan2(self.ey, self.ex) + np.pi

        # .parameters from opts.csv
        keys = [
            "time_step",
            "thickness",
            "division_length_1",
            "division_length_2",
            "diffusion_constant",
            "force_constant",
            "initial_type",
            "growth_rate_1",
            "growth_rate_2",
            "rng_seed",
            "initial_cells",
            "system_size",
        ]

        # .add legacy keys
        legacy = {
            "division_length_1": "division_lengths_1",
            "division_length_2": "division_lengths_2",
            "growth_rate_1": "growth_rates_1",
            "growth_rate_2": "growth_rates_2",
        }

        for key in keys:
            if key in opts:
                setattr(self, key, opts[key])
            elif legacy[key] in opts:
                setattr(self, key, opts[legacy[key]])

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
