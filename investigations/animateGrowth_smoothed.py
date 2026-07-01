from functools import wraps, partial
import multiprocessing.pool
import os
import argparse
import multiprocessing
import pathlib
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from tools.graphics.snapshots_smoothed import drawRods, gridSnapshots, flowfield
from tools.dataAPI.datamodel import DataModel
from routines.parameterTable import parameterTable
from tools.dataModels.measurements import Measurements
import importlib
import warnings


warnings.simplefilter("ignore", OptimizeWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="workspace/simulations/")
parser.add_argument("--img_path", default="")
parser.add_argument("--n_cores", default=4, type=int)
parser.add_argument("--existing", action="store_true")
parser.add_argument("--grid_layout", action="store_true")
parser.add_argument("--flows", action="store_true")
parser.add_argument(
    "--colorby",
    default="population",
    choices=[
        "angle",
        "population",
        "smectic",
        "heterozygosity",
        "nematic",
        "splits",
        "radial",
        "length",
        "flows"
    ],
)
args = parser.parse_args()

inputPath = args.input

colorOptions = [
    "angle",
    "population",
    "smectic",
    "heterozygosity",
    "nematic",
    "splits",
    "radial",
    "length",
    "flows"
]


df = parameterTable(inputPath)

def extract_data(dataPath: str, time: int):
    data = Measurements(dataPath=dataPath, time=time)
    return data


if args.colorby not in colorOptions:
    raise ValueError(f"colorby should be one of {colorOptions}")


if args.grid_layout:
    ext = "grid_"
else:
    ext = args.colorby + "_"

df = parameterTable(inputPath)


def exponential_func(t, a, b, c):
    return a * np.exp(b * t) + c


for (i, subpath) in enumerate(df.basePath):
    print(subpath)
    opts, times = DataModel.extract_times(subpath)

    if args.img_path != "":
        simpath = subpath.split("/")[-1]
        imgPath = os.path.join(args.img_path, simpath, "images/")
    else:
        imgPath = os.path.join(subpath, "images/")

    pathlib.Path(imgPath).mkdir(parents=True, exist_ok=True)

    radii = []
    for time in times:
        data = extract_data(subpath, time)
        radius = (max(data.x) ** 2 + max(data.y) ** 2) ** 0.5
        radii.append(radius)
        print(f"Timestep {time}: Colony Radius = {radius:.2f}")
        last_time = times[-1]

    
    try:
        scaled_times = np.array(times) / 100000  
        initial_guess = [1.0, 0.1, 1.0]  
        popt, pcov = curve_fit(exponential_func, scaled_times, radii, p0=initial_guess, bounds=(0, [np.inf, np.inf, np.inf]))
        a, b, c = popt
        

        
        fitted_radii = exponential_func(scaled_times, *popt)
        

        
        fitted_radii_dict = {time: radius for time, radius in zip(times, fitted_radii)}
        fitted_radii_dict[-1] = fitted_radii[-1]
        

    except RuntimeError as e:
        print("Error in curve fitting:", e)

    
    if not args.existing:
        if args.grid_layout:
            task = partial(
                gridSnapshots, fitted_radii_dict=fitted_radii_dict, datapath=subpath, imgPath=imgPath, saveImg=True
            )
        elif args.colorby == "flows":
            task = partial(
                flowfield, datapath=subpath, imgPath=imgPath, saveImg=True, pxNumber=100
            )
        else:
            task = partial(
                drawRods,
                fitted_radii_dict=fitted_radii_dict,
                datapath=subpath,
                imgPath=imgPath,
                hidedecoration=True,
                saveImg=True,
                colorby=args.colorby 
            )

        with multiprocessing.Pool(args.n_cores) as pool:
            pool.map(task, times)

        
        for i in range(len(times)):
            savefrom = f"{imgPath}/{times[i]}.png"
            saveto = f"{imgPath}/final_{i}.png"
            os.rename(savefrom, saveto)

    
    images = [f"final_{i}.png" for i in range(len(times))]
    video_name = imgPath + f"/{ext}animation.mp4"

    
    if (spec := importlib.util.find_spec("cv2")) is not None:
        import cv2
        frame = cv2.imread(os.path.join(imgPath, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        video = cv2.VideoWriter(
            filename=video_name, fourcc=fourcc, fps=6, frameSize=(width, height)
        )

        for image in images:
            video.write(cv2.imread(os.path.join(imgPath, image)))

        cv2.destroyAllWindows()
        video.release()
    elif (spec := importlib.util.find_spec("imageio")) is not None:
        import imageio
        video_name = imgPath + f"/{ext}animation.gif"
        ims = [imageio.v2.imread(os.path.join(imgPath, image)) for image in images]
        imageio.mimwrite(video_name, ims)
    else:
        pass
