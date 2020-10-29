import argparse, os, subprocess, glob, json
import pims
import skimage
from skimage import io
from skimage.transform import resize
from PIL import Image
import numpy as np

# https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html

MANGA_SIZE = 256
EXT = "png"

def save_mp4(path: str, i: int, ext: str) -> None:
    """ Saves data into the mp4 format. """
    folder = "/".join(path.split("/")[:-1])
    subprocess.call(["ffmpeg", "-i", path, "-codec", "copy", f"{folder}/{i}.{ext}"])

def change_ext(args) -> None:
    """ Runs ffmpeg on each file in a specificed folder. """
    for i, path in enumerate(sorted(glob.glob(args.path))):
        save_mp4(path, i, args.format)

def process_manga(args):
    frames = pims.open(args.path)
    folder = "/".join(args.path.split("/")[:-1])
    config = json.load(open(f"{folder}/config.json"))
    # crop whitespace from the edges
    # top, bottom, left, right, color things
    box = config["crop"]
    frames = pims.process.crop(frames, ((box[0], box[1]), (box[2], box[3]), (0, 0)))
    # apply greyscale transformation
    frames = pims.as_grey(frames)
    # resize images
    size = pims.pipeline(lambda img: resize(img, (MANGA_SIZE, MANGA_SIZE)))
    frames = size(frames)

    # convert image to unsigned int, avoid rounding error by using image max
    scale = lambda img: skimage.img_as_ubyte(img/max(255, np.max(img)))
    frames = pims.pipeline(scale)(frames)

    # show image
    # Image.fromarray(frames[14]).show()

    # create folders with the same structure if they don't exist
    data_folder = folder.replace("data", "preprocess")
    os.makedirs(data_folder, exist_ok=True)

    # remove old images
    for fname in glob.glob(f"{data_folder}/*.{EXT}"):
        os.remove(fname)

    for i in range(config.get("start", 0), len(frames) - config.get("end", 0)):
        if i not in config.get("exclude", []):
            io.imsave(f"{data_folder}/{i}.{EXT}", frames[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data manipulation.")
    parser.add_argument("-v", "--version", action="version", version="data 1.0")

    subparsers = parser.add_subparsers(title="commands")
    reformat = subparsers.add_parser("reformat", help="generates mp4 files from mkv")
    reformat.add_argument("-f", "--format", help="target format", default="mp4")
    reformat.add_argument("-p", "--path", help="glob containing mkv files")
    reformat.set_defaults(func=change_ext)

    manga = subparsers.add_parser("manga", help="preprocess a manga folder")
    manga.add_argument("-p", "--path", help="folder containing a series of PNGs")
    manga.set_defaults(func=process_manga)

    args = parser.parse_args()
    if "func" in args:
        args.func(args)

