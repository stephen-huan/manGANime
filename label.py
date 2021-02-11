import os, json, argparse, bisect, math
import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import pims
import numpy as np

DATA = "data" # data folder
ANIME, MANGA = f"{DATA}/anime", f"{DATA}/manga"
CUTOFF = 0.8 # cutoff for dot product idenitfication of cuts 

if os.path.exists("config.json"):
    with open("config.json") as f:
        config = json.load(f)
else:
    config = {}

def set_image(label: tk.Label, img: Image, scale: float=0.5) -> None:
    """ Change which image a label shows. """
    x, y = img.size
    # upscale for preprocessed, resized images
    if max(x, y) < 600:
        img = img.resize((int(x/scale), int(y/scale)))
    else:
        img.thumbnail((int(x*scale), int(y*scale)))
    render = ImageTk.PhotoImage(img)
    label.configure(image=render)
    label.image = render

def set_frame(entry: tk.Entry, i: int) -> None:
    """ Change which text an entry shows. """
    entry.delete(0, tk.END)
    entry.insert(0, str(i + 1))

def bound(i: int, n: int):
    """ Return a number between 0 and n. """
    return max(min(i, n), 0)

def norm(m: np.array) -> float:
    """ Magnitude of a matrix. """
    return np.sqrt(np.sum(m*m))

def dot(u: np.array, v: np.array) -> float:
    """ Dot product between two matrices. """
    u, v = u.astype(np.float), v.astype(np.float)
    return np.sum(u*v)/(norm(u)*norm(v))

class Window(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        # path controls
        self.anime_btn = tk.Button(self, text="anime_path: N/A",
                               command=self.change_anime)
        self.anime_btn.grid(row=0, column=0)
        self.manga_btn = tk.Button(self, text="manga_path: N/A",
                               command=self.change_manga)
        self.manga_btn.grid(row=1, column=0)
        # save buttons
        self.save_anime_btn = tk.Button(self, text="save frame",
                                        command=self.save("anime"))
        self.save_anime_btn.grid(row=0, column=1)
        self.save_manga_btn = tk.Button(self, text="save page",
                                        command=self.save("manga"))
        self.save_manga_btn.grid(row=1, column=1)
        # tagging button
        self.tag_btn = tk.Button(self, text="tag", command=self.tag)
        self.tag_btn.grid(row=0, column=3)
        # cut jump buttons
        self.cut_btn = tk.Button(self, text="prev chunk", command=lambda: self.cut(-1))
        self.cut_btn.grid(row=0, column=2)
        self.cut_btn = tk.Button(self, text="next chunk", command=lambda: self.cut(1))
        self.cut_btn.grid(row=1, column=2)

        # image displays
        self.manga_img = tk.Label(self)
        self.manga_img.grid(row=2, column=0, columnspan=4, sticky=tk.W + tk.E, ipadx=10)
        self.anime_img = tk.Label(self)
        self.anime_img.grid(row=2, column=5, columnspan=4)
        # default images
        set_image(self.anime_img, Image.open(f"images/anime_filler.png"))
        set_image(self.manga_img, Image.open(f"images/manga_filler.png"))

        # movement controls
        self.manga_left  = tk.Button(self, text="<", command=self.adj("manga", -1))
        self.manga_left.grid(row=3, column=0, sticky=tk.E)
        self.manga_right = tk.Button(self, text=">", command=self.adj("manga", 1))
        self.manga_right.grid(row=3, column=3, sticky=tk.W)
        self.manga_box = tk.Entry(self, width=3)
        self.manga_box.bind("<Return>", lambda event: self.jump("manga"))
        self.manga_box.insert(0, "???")
        self.manga_box.grid(row=3, column=1)
        self.manga_frame = tk.Label(self, text="/???")
        self.manga_frame.grid(row=3, column=2)

        self.anime_left  = tk.Button(self, text="<", command=self.adj("anime", -1))
        self.anime_left.grid(row=3, column=5, sticky=tk.E)
        self.anime_right = tk.Button(self, text=">", command=self.adj("anime", 1))
        self.anime_right.grid(row=3, column=8, sticky=tk.W)
        self.anime_box = tk.Entry(self, width=5)
        self.anime_box.insert(0, "???")
        self.anime_box.bind("<Return>", lambda event: self.jump("anime"))
        self.anime_box.grid(row=3, column=6)
        self.anime_frame = tk.Label(self, text="/???")
        self.anime_frame.grid(row=3, column=7)
        self.anime_step = tk.Entry(self, width=5)
        self.anime_step.insert(0, "1")
        self.anime_step.grid(row=3, column=8)

        if args.check is not None or "anime_path" in config:
            self.change_anime()

    def move(self, name: str, i: int):
        """ Change the position of the viewer. """
        try:
            stream = getattr(self, name)
            setattr(self, name + "_i", i)
            set_image(getattr(self, name + "_img"), Image.fromarray(stream[i]))
            set_frame(getattr(self, name + "_box"), i)
        except AttributeError:
            print("Media has not been loaded!")
        else:
            if args.check is not None and name == "anime" and args.interpolate:
                # if tag exists, use tag
                k = bisect.bisect_left(self.tags, i)
                if self.tags[k] < len(self.anime) and i <= self.tags[k]:
                    j = k
                # otherwise, set manga page to linearly interpolated guess
                else:
                    k = bisect.bisect_left(self.tags, len(self.anime))
                    frames = self.tags[k - 1]
                    if frames != len(self.anime):
                        r = (len(self.manga) - k)/(len(self.anime) - 1 - frames)
                        if r != self.ratio and r != 0:
                            print(f"new ratio: {1/r:.3f}")
                            self.ratio = r
                    else:
                        frames, k = 0, 0
                    j = k + math.floor((i - frames)*self.ratio)

                self.move("manga", min(j, len(self.manga) - 1))

    def adj(self, name: str, d: int):
        """ Implement < and > control buttons. """
        def func():
            stream = getattr(self, name)
            step = int(self.anime_step.get()) if name == "anime" else 1
            i = bound(getattr(self, name + "_i") + d*step, len(stream) - 1)
            self.move(name, i)
        return func

    def jump(self, name: str):
        """ Implement typing a frame number into a textbox."""
        stream = getattr(self, name)
        try:
            i = bound(int(getattr(self, name + "_box").get()) - 1, len(stream) - 1)
            self.move(name, i)
        except ValueError:
            print("Not a number!")

    def save(self, name: str):
        """ Saves the current frame. """
        def func():
            stream, i = getattr(self, name), getattr(self, name + "_i")
            Image.fromarray(stream[i]).save(f"{name}_img.png")
        return func

    def change_anime(self):
        """ Load an anime. """
        if args.check is not None:
            # load manga
            self.change_manga()
            path = anime_path
            self.anime = pims.open(path)
            # exclude certain ranges of indexes 
            exclude = set(x for r in anime_config["exclude"]
                          for x in range(r[0], r[1] + 1))
            indexes = sorted(set(range(len(self.anime))) - set(exclude))
            self.anime = self.anime[indexes]
            self.ratio = (len(self.manga) - 1)/(len(self.anime) - 1)
            print(f"frames per page: {1/self.ratio:.3f}")
            # tags
            if os.path.exists(anime_folder + f"/{anime_name}.json"):
                print("loaded tags from file")
                with open(anime_folder + f"/{anime_name}.json") as f:
                    self.tags = json.load(f)
            else:
                self.tags = [len(self.anime)]*len(self.manga)
        else:
            print(f"Select an anime file")
            path = config["anime_path"] if "anime_path" in config else \
                fd.askopenfilename(initialdir=DATA)
            print(f"The file you selected was: {path}")
            self.anime = pims.open(path)
        self.anime_path = path
        self.anime_btn["text"] = f"anime_path: {path}"
        self.anime_i = 0
        set_image(self.anime_img, Image.fromarray(self.anime[0]))
        set_frame(self.anime_box, 0)
        self.anime_frame["text"] = f"/{len(self.anime)}"

    def change_manga(self):
        """ Load a manga. """
        if args.check is not None:
            path = anime_config["manga"]
            self.manga = pims.open(f"preprocess/manga/{path}/*.png")
            i, j = anime_config["pages"]
            self.manga = self.manga[i: j + 1]
        else:
            print(f"Select an manga folder")
            path = config["manga_path"] if "manga_path" in config else \
                fd.askdirectory(initialdir=DATA)
            print(f"The folder you selected was: {path}")
            # load a series of sequential images as a pims object
            self.manga = pims.open(f"{path}/*.png")
        self.manga_path = path
        self.manga_btn["text"] = f"manga_path: {path}"
        self.manga_i = 0
        set_image(self.manga_img, Image.fromarray(self.manga[0]))
        set_frame(self.manga_box, 0)
        self.manga_frame["text"] = f"/{len(self.manga)}"

    def tag(self):
        """ A tag is a list of indexes, where the index is a manga page and
        the value is the largest anime frame which maps to the manga page. """
        self.tags[self.manga_i] = self.anime_i

        # save tags
        with open(anime_folder + f"/{anime_name}.json", "w") as f:
            json.dump(self.tags, f)

    def cut(self, delta):
        """ Find the next cut by looking at the dot product between adjacent
        frames, declaring a cut if the similarity is below a certain cutoff. """
        i = self.anime_i
        while 0 <= i < len(self.anime) - 1:
            im1, im2 = self.anime[i], self.anime[i + delta]
            if dot(im1, im2) <= CUTOFF:
                # normally put viewer before cut, but if pressed at cut skip
                if i == self.anime_i:
                    i += delta
                self.move("anime", i)
                return
            i += delta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data visualization.")
    parser.add_argument("-v", "--version", action="version", version="tag 1.0")
    parser.add_argument("-c", "--check", help="tag a file")
    parser.add_argument("-i", "--interpolate", action="store_true", help="linearly interpolate non-tagged portions")
    args = parser.parse_args()

    if args.check is not None:
        anime_path = args.check
        anime_name = args.check.split("/")[-1].split(".")[0]
        anime_folder = "/".join(anime_path.split("/")[:-1])
        with open(anime_folder + "/config.json") as f:
            anime_config = json.load(f)
        anime_config = anime_config[anime_name]

    root = tk.Tk()
    app = Window(root)
    root.wm_title("Data Tagger")
    root.geometry("1920x1080")
    root.mainloop()

