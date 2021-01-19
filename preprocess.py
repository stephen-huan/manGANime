import argparse, os, subprocess, glob, json
import pims
import av
import skimage
from skimage import io
from skimage.transform import resize
from PIL import Image
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

MANGA_SIZE = 256
MANGA_EXT  = "png"
ANIME_SIZE = 256
ANIME_EXT  = "mp4"
ANIME_FPS  = 24
DURATION = 3*60*ANIME_FPS # search the first five minutes for an intro

def save_mp4(path: str, i: int, ext: str) -> None:
    """ Saves data into the mp4 format. """
    folder = "/".join(path.split("/")[:-1])
    subprocess.call(["ffmpeg", "-i", path, "-codec", "copy", f"{folder}/{i}.{ext}"])

def change_size(path: str, dest: str) -> None:
    """ Changes the video to ANIME_SIZE x ANIME_SIZE.  """
    subprocess.call(["ffmpeg", "-i", path, "-vf", f"scale={ANIME_SIZE}:{ANIME_SIZE},setsar=1:1",
                     "-sws_flags", "bicubic", dest])

def change_ext(args) -> None:
    """ Runs ffmpeg on each file in a specificed folder. """
    for i, path in enumerate(sorted(glob.glob(args.path))):
        save_mp4(path, i, args.format)

def save_video(fname: str, video, fps: int=ANIME_FPS) -> None:
    """ Saves a pims video as a mp4 file.
    https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html """
    fourcc = cv.VideoWriter_fourcc(*"avc1")
    out = cv.VideoWriter(fname, fourcc, fps, video[0].shape[:2][::-1])
    for img in video:
        out.write(cv.cvtColor(img, cv.COLOR_RGB2BGR))
    out.release()

@pims.pipeline
def to_ubyte(img: np.array):
    """ Converts an image to a integer array between 0 and 255. """
    # convert image to unsigned int, avoid rounding error by clipping
    return skimage.img_as_ubyte(np.clip(img/255, 0, 1))

def transform(transformations: list, frames):
    """ Apply transformations to a given pims sequence. """
    for t in transformations:
        frames = t(frames)
    return frames

def transform_manga(config: dict, frames):
    """ Apply transformations to a sequence of manga pages. """
    # crop whitespace from the edges
    # top, bottom, left, right, color things
    box = config["crop"]
    trans = [lambda frames: pims.process.crop(frames,
                            ((box[0], box[1]), (box[2], box[3]), (0, 0))),
             # apply greyscale transformation
             pims.as_grey,
             # resize images
             pims.pipeline(lambda img: resize(img, (MANGA_SIZE, MANGA_SIZE))),
             to_ubyte
            ]
    return transform(trans, frames)

def preprocess_anime(global_config: dict, video_config: dict, video):
    """ Apply initial transformations to an anime video. """
    # exclude certain ranges of indexes 
    exclude = set(x for r in video_config["exclude"]
                  for x in range(r[0], r[1] + 1))
    indexes = sorted(set(range(len(video))) - set(exclude))
    # take every stride-th frame
    video = video[indexes][::global_config["stride"]]

    return video

def pims_transform_anime(global_config: dict, video_config: dict, video):
    """ Apply transformations to an anime video, using pims pipelines. """
    video = preprocess_anime(global_config, video_config, video)
    trans = [pims.pipeline(lambda img: resize(img, (ANIME_SIZE, ANIME_SIZE))),
             pims.pipeline(skimage.img_as_ubyte)
            ]
    return transform(trans, video)

def torch_transform_anime(global_config: dict, video_config: dict, video):
    """ Apply transformations to an anime video, using torch methods. """
    video = preprocess_anime(global_config, video_config, video)
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((ANIME_SIZE, ANIME_SIZE)),
    ])
    return [np.asarray(trans(frame)) for frame in video]

def pillow_transform_anime(global_config: dict, video_config: dict, video):
    """ Apply transformations to an anime video, using pillow methods. """
    video = preprocess_anime(global_config, video_config, video)
    return [np.asarray(Image.fromarray(frame).resize((ANIME_SIZE, ANIME_SIZE)))
            for frame in video]

transform_anime = torch_transform_anime # which backend to use

def process_manga(args):
    """ Preprocess manga by cropping, grescaling, and resizing. """
    folder = "/".join(args.path.split("/")[:-1])
    config = json.load(open(f"{folder}/config.json"))
    # create folders with the same structure if they don't exist
    data_folder = folder.replace("data", "preprocess")
    os.makedirs(data_folder, exist_ok=True)
    # remove old images
    for fname in glob.glob(f"{data_folder}/*.{MANGA_EXT}"):
        os.remove(fname)

    frames = transform_manga(config, pims.open(args.path))
    # show image
    # Image.fromarray(frames[14]).show()

    for i in range(config.get("start", 0), len(frames) - config.get("end", 0)):
        if i not in config.get("exclude", []):
            io.imsave(f"{data_folder}/{i}.{MANGA_EXT}", frames[i])

def process_anime(args):
    """ Preprocess anime by resizing. """
    folder = "/".join(args.path.split("/")[:-1])
    config = json.load(open(f"{folder}/config.json"))
    # create folders with the same structure if they don't exist
    data_folder = folder.replace("data", "preprocess")
    os.makedirs(data_folder, exist_ok=True)
    # remove old videos
    for fname in glob.glob(f"{data_folder}/*.{ANIME_EXT}"):
        os.remove(fname)

    for fname in glob.glob(args.path):
        name = fname.split("/")[-1].split(".")[0]
        # video = transform_anime(config, config[name], pims.open(fname))
        # save_video(f"{data_folder}/{name}.{ANIME_EXT}", video)
        change_size(fname, f"{data_folder}/{name}.{ANIME_EXT}")

def find_intro(args):
    """ Finds the spot where the image occurs in the video. """
    folder = "/".join(args.path.split("/")[:-1])
    config = json.load(open(f"{folder}/config.json"))
    t, info = args.type == "intro", config[args.type]
    l, r = info["left"], info["right"]
    img = np.asarray(Image.open(f"{folder}/{info['image']}")).astype(np.float)
    for fname in glob.glob(args.path):
        name = fname.split("/")[-1].split(".")[0]
        video = pims.open(fname)
        best, besti = -float("inf"), 0
        for i, frame in enumerate(video[:DURATION] if t else video[-DURATION:]):
            frame = frame.astype(np.float)
            # dot product between two matrices, flattened to a vector
            k = np.sum(img*frame)/np.sqrt(np.sum(frame*frame))
            if k > best:
                best, besti = k, i
        besti += 0 if t else len(video) - DURATION
        Image.fromarray(video[besti]).show()
        print(f"cosine similarity: {best/np.sqrt(np.sum(img*img)):.3f}")
        ans = input(f"correct for {fname}? (y/n) ")
        if ans[0].lower() == "y":
            if name not in config:
                config[name] = {"exclude": []}
            config[name]["exclude"].append([besti - l, min(besti + r, len(video))])

    with open(f"{folder}/config.json", "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data manipulation.")
    parser.add_argument("-v", "--version", action="version", version="data 1.0")

    subparsers = parser.add_subparsers(title="commands")
    reformat = subparsers.add_parser("reformat", help="generates mp4 files from mkv")
    reformat.add_argument("-f", "--format", help="target format", default="mp4")
    reformat.add_argument("-p", "--path", help="glob containing mkv files")
    reformat.set_defaults(func=change_ext)

    manga = subparsers.add_parser("manga", help="preprocess a manga folder")
    manga.add_argument("-p", "--path", help="folder containing a series of image files")
    manga.set_defaults(func=process_manga)

    anime = subparsers.add_parser("anime", help="preprocess an anime folder")
    anime.add_argument("-p", "--path", help="folder containing a series of video files")
    anime.set_defaults(func=process_anime)

    search = subparsers.add_parser("search", help="remove intro/outros")
    search.add_argument("-p", "--path", help="folder containing a series of video files")
    search.add_argument("-t", "--type", help="intro or outro")
    search.set_defaults(func=find_intro)

    args = parser.parse_args()

    # run on multiple GPUs if possible, defaulting to CPU
    CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if CUDA else "cpu")
    if CUDA:
        print(f"torch using {torch.cuda.device_count()} GPUs")
    else:
        print("torch running on CPU")

    if "func" in args:
        args.func(args)

