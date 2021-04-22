import argparse, pickle
import click
import numpy as np
from PIL import Image
import pims
import cv2 as cv
import torch
from stylegan2_ada_pytorch import projector, generate

torch.manual_seed(0) # set seed
FPS = 24             # frames per second
OUT = "out"          # output directory
N, M = 1, 8          # size of image grid

def save_video(fname: str, video, fps: int=FPS) -> None:
    """ Saves a pims video as a mp4 file.
    https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html """
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(fname, fourcc, fps, video[0].shape[:2][::-1])
    for img in video:
        out.write(cv.cvtColor(img, cv.COLOR_RGB2BGR))
    out.release()

def laod_image(fname: str) -> np.array:
    """ Loads an image from disk. """
    return np.array(Image.open(f"{OUT}/{fname}"))

def summarize(args, n: int=N, m: int=M) -> None:
    """ Generates a summary grid of the video. """
    video = pims.open(args.path)
    n, t = len(video) - 1, N*M
    frames = [video[round((i/(t - 1))*n)] for i in range(t)]
    h, w = frames[0].shape[:-1]
    out = Image.new("RGB", (M*w, N*h))
    for i in range(t):
        x, y = i//M, i % M
        out.paste(Image.fromarray(frames[i]), (w*y, h*x))
    out.save(f"{OUT}/grid.png")

def gen_image(w: np.array) -> np.array:
    """ Given a latent code, generates an image. """
    np.savez("temp.npz", w=w)
    generate.generate_images(["--network", args.network, "--outdir", OUT,
                              "--projected-w", "temp.npz"])
    return load_image("proj00.png")

def project_image(im: np.array) -> tuple:
    """ Gets the latent code and resulting image associated with an image. """
    Image.fromarray(im).save("temp.png")
    projector.run_projection(["--network", args.network, "--target", "temp.png",
                              "--outdir", OUT, "--save-video", False,
                              "--num-steps", 100])
    return load_image("proj.png"), np.load(f"{OUT}/projected_w.npz")["w"]

def project_video(args) -> None:
    """ Projects each frame of a video into the latent space. """
    video = pims.open(args.path)
    out = [project_image(frame)[0] for frame in video]
    save_video(f"{OUT}/proj.mp4", out)

def lerp(args) -> None:
    """ Linearly interpolates to reconstruct a video. """
    video = pims.open(args.path)
    start, end = project_image(video[0])[1], project_image(video[-1])[1]
    n, d = len(video), end - start
    out = [gen_image(start + (t/(n - 1))*d) for t in range(n)]
    save_video(f"{OUT}/lerp.mp4", out)

if __name__ == "__main__":
    commands = {
        "project": project_video,
        "lerp": lerp,
        "summarize": summarize,
    }

    parser = argparse.ArgumentParser(description="Path interpolation.")
    parser.add_argument("-v", "--version", action="version", version="path 1.0")
    parser.add_argument("-p", "--path", help="path to video file")
    parser.add_argument("-n", "--network", help="path to saved network")
    parser.add_argument("command", choices=commands.keys(), help="action to take")
    args = parser.parse_args()

    commands[args.command](args)

