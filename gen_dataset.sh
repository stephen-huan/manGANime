#!/bin/bash

# generate a list of pngs for stylegan
# fps is frames TAKEN per second, not the "skip" number of frames
# use 6 fps (take every 4th frame) because of heavy redundancy
# or use 12 fps (take every other frame) becuase of "on twos", ~180,000 images
# could automate by looking at metadata --- "Episode" title ... but no need

ffmpeg -i  preprocess/anime/glt/0.mp4 -ss 00:02:38 -to 00:23:42 -vf fps=$1 image00-%05d.png
# move to folder after every command because of the sheer number of frames
# will run out of memory otherwise
mv *.png preprocess/source
ffmpeg -i  preprocess/anime/glt/1.mp4 -ss 00:02:56 -to 00:22:04 -vf fps=$1 image01-%05d.png
mv *.png preprocess/source
ffmpeg -i  preprocess/anime/glt/2.mp4 -ss 00:01:30 -to 00:22:04 -vf fps=$1 image02-%05d.png
mv *.png preprocess/source
ffmpeg -i  preprocess/anime/glt/3.mp4 -ss 00:01:54 -to 00:22:03 -vf fps=$1 image03-%05d.png
mv *.png preprocess/source
ffmpeg -i  preprocess/anime/glt/4.mp4 -ss 00:01:30 -to 00:21:27 -vf fps=$1 image04-%05d.png
mv *.png preprocess/source
ffmpeg -i  preprocess/anime/glt/5.mp4 -ss 00:01:30 -to 00:21:54 -vf fps=$1 image05-%05d.png
mv *.png preprocess/source
ffmpeg -i  preprocess/anime/glt/6.mp4 -ss 00:02:39 -to 00:21:30 -vf fps=$1 image06-%05d.png
mv *.png preprocess/source
ffmpeg -i  preprocess/anime/glt/7.mp4 -ss 00:02:12 -to 00:22:15 -vf fps=$1 image07-%05d.png
mv *.png preprocess/source
ffmpeg -i  preprocess/anime/glt/8.mp4 -ss 00:00:55 -to 00:22:04 -vf fps=$1 image08-%05d.png
mv *.png preprocess/source
ffmpeg -i  preprocess/anime/glt/9.mp4 -ss 00:02:30 -to 00:21:59 -vf fps=$1 image09-%05d.png
mv *.png preprocess/source
ffmpeg -i preprocess/anime/glt/10.mp4 -ss 00:01:57 -to 00:22:10 -vf fps=$1 image10-%05d.png
mv *.png preprocess/source
ffmpeg -i preprocess/anime/glt/11.mp4 -ss 00:00:33 -to 00:22:18 -vf fps=$1 image11-%05d.png
mv *.png preprocess/source

# create zip
python stylegan2-ada-pytorch/dataset_tool.py --source preprocess/source --dest stylegan.zip

