import os, json
import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import pims
import numpy as np

DATA = "data"
ANIME, MANGA = f"{DATA}/anime", f"{DATA}/manga"

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

    def move(self, name: str, i: int):
        """ Change the position of the viewer. """
        try:
            stream = getattr(self, name)
            setattr(self, name + "_i", i)
            set_image(getattr(self, name + "_img"), Image.fromarray(stream[i]))
            set_frame(getattr(self, name + "_box"), i)
        except AttributeError:
            print("Media has not been loaded!")

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
        print(f"Select an anime file")
        path = config["anime_path"] if "anime_path" in config else \
            fd.askopenfilename(initialdir=DATA)
        print(f"The file you selected was: {path}")
        self.anime_path = path
        self.anime_btn["text"] = f"anime_path: {path}"
        self.anime = pims.open(path)
        self.anime_i = 0
        set_image(self.anime_img, Image.fromarray(self.anime[0]))
        set_frame(self.anime_box, 0)
        self.anime_frame["text"] = f"/{len(self.anime)}"

    def change_manga(self):
        """ Load a manga. """
        print(f"Select an manga folder")
        path = config["manga_path"] if "manga_path" in config else \
            fd.askdirectory(initialdir=DATA)
        print(f"The folder you selected was: {path}")
        self.manga_path = path
        self.manga_btn["text"] = f"manga_path: {path}"
        # load a series of sequential images as a pims object
        self.manga = pims.open(f"{path}/*.png")
        self.manga_i = 0
        set_image(self.manga_img, Image.fromarray(self.manga[0]))
        set_frame(self.manga_box, 0)
        self.manga_frame["text"] = f"/{len(self.manga)}"

if __name__ == "__main__":
    root = tk.Tk()
    app = Window(root)
    root.wm_title("Data Tagger")
    root.geometry("1920x1080")
    root.mainloop()

