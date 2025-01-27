from functools import wraps, partial
import multiprocessing.pool
import os
import argparse
import multiprocessing
import pathlib
from tools.graphics.snapshots import drawRods, gridSnapshots, flowfield
from tools.dataAPI.datamodel import DataModel
from routines.parameterTable import parameterTable
import importlib

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

if not args.colorby in colorOptions:
    raise ValueError(f"colorby should be one of {colorOptions}")

if args.grid_layout:
    ext = "grid_"
else:
    ext = args.colorby + "_"

df = parameterTable(inputPath)

msg = "image and video generation successful"

# .collect all experiments
for (i, subpath) in enumerate(df.basePath):
    print(subpath)
    opts, times = DataModel.extract_times(subpath)

    if args.img_path != "":
        simpath = subpath.split("/")[-1]
        imgPath = os.path.join(args.img_path, simpath, "images/")
    else:
        imgPath = os.path.join(subpath, "images/")

    pathlib.Path(imgPath).mkdir(parents=True, exist_ok=True)

    if not args.existing or len(os.listdir(imgPath)) == 0:
        if args.grid_layout:
            task = partial(
                gridSnapshots, datapath=subpath, imgPath=imgPath, saveImg=True
            )
        elif args.colorby == "flows":
            task = partial(
                flowfield, datapath=subpath, imgPath=imgPath, saveImg=True, pxNumber=100
            )
        else:
            task = partial(
                drawRods,
                datapath=subpath,
                imgPath=imgPath,
                hidedecoration=True,
                saveImg=True,
                colorby=args.colorby,
            )

        with multiprocessing.Pool(args.n_cores) as pool:
            pool.map(task, times)

        # .rename files
        for i in range(len(times)):
            savefrom = f"{imgPath}/{times[i]}.png"
            saveto = f"{imgPath}/final_{i}.png"
            os.rename(savefrom, saveto)

    # .create video from images
    images = [f"final_{i}.png" for i in range(len(times))]
    video_name = imgPath + f"/{ext}animation.mp4"

    if len(images) == 0:
        msg += "\nNo images to create video for {}".format(subpath)
        continue

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
        msg = "Neither cv2 nor imageio is installed. Cannot create video."
        pass

print(msg)