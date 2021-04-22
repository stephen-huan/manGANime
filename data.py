import json, bisect
import numpy as np
from PIL import Image
import pims
import torch
from torch.utils import data

torch.manual_seed(0)          # set seed
PATH = "preprocess/anime/glt" # folder containing video files and json config
MANGA = "preprocess/manga"    # folder containing manga
VIDEO_EXT = "mp4"             # format for anime
IMAGE_EXT = "png"             # format for manga

class Dataset(data.Dataset):

    """ Custom dataset containing manga pages and anime frames. """

    def __init__(self, path: str,
                 stride: int, window: int, num_input: int, num_output: int):
        self.path, self.stride, self.window, self.num_input, self.num_output = \
             path,      stride,      window,      num_input,      num_output
        with open(path + "/config.json") as f:
            self.config = json.load(f)

        self.anime, self.manga, self.tags = [], [], []
        eps = sorted(filter(lambda x: x.isdigit(), self.config),
                     key=lambda x: int(x))
        for fname in eps:
            conf = self.config[fname]
            if not conf.get("include", True):
                continue
            # load anime
            anime = pims.open(f"{path}/{fname}.{VIDEO_EXT}")
            exclude = set(x for r in conf["exclude"]
                          for x in range(r[0], r[1] + 1))
            indexes = sorted(set(range(len(anime))) - set(exclude))
            self.anime.append(anime[indexes][::stride])
            # load manga
            manga = pims.open(f"{MANGA}/{conf['manga']}/*.{IMAGE_EXT}")
            i, j = conf["pages"]
            self.manga.append(manga[i: j + 1])
            # load tags
            with open(f"{path}/{fname}.json") as f:
                self.tags.append(json.load(f))

        # global index to epsisode mapping
        lens = [len(ep) - num_output - num_input for ep in self.anime]
        self.len = sum(lens)
        # compute prefix sum for indexing
        self.anime_lens = [0]*(len(lens) + 1)
        for i in range(len(lens)):
            self.anime_lens[i + 1] = self.anime_lens[i] + lens[i]

    def __len__(self) -> int: return self.len

    def __getitem__(self, index: int) -> tuple:
        """ Get data instance. """
        ep = bisect.bisect(self.anime_lens, index) - 1
        anime, manga, tags = self.anime[ep], self.manga[ep], self.tags[ep]
        index -= self.anime_lens[ep]
        # input tensor: last num_input frames as well as window of manga pages 
        prev = [anime[i] for i in range(index, index - self.num_input, -1)]
        page = bisect.bisect_left(tags, self.stride*index)
        pages = [manga[i] if 0 <= i < len(manga) else np.zeros(manga.frame_shape)
                 for i in range(page - self.window, page + self.window + 1)]
        # reshape 3x256x256 manga pages to 256x256x3 tensor like image 
        page_tensor = np.array(pages).reshape(manga.frame_shape + (3,))
        X = torch.tensor(prev + [page_tensor])
        # output tensor: next num_output frames
        out = [anime[i] for i in range(index, index + self.num_output)]
        y = torch.tensor(out)

        # for frame in prev:
        #     Image.fromarray(frame).show()
        # for page in pages:
        #     Image.fromarray(page).show()
        # input()

        return X, y

params = {
    "stride": 2,     # how many frames to skip
    "window": 1,     # number of manga pages before/after 
    "num_input": 1,  # how many frames to condition model on
    "num_output": 1, # how many frames to output
}
train = Dataset(PATH, **params)
load_params = {
    "batch_size": 64,
    "shuffle": True,
    # "num_workers": 10
}
train_gen = data.DataLoader(train, **load_params)

if __name__ == "__main__":
    for batch, labels in train_gen:
        print(batch, labels)
        break

