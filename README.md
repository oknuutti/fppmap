# Fully perspective projected map (fppmap)

## Summary
Overlay a map on an image. The map is projected on the image using the camera parameters and pose given.

## Setup
Setting up environment using conda:
```
git clone https://github.com/oknuutti/fppmap.git fppmap
cd ffpmap
conda env create -f environment.yml
```

## Example
Run by e.g.:
```
python main.py --image myimg.png --fov 43.6 --width 2048 --height 1536 \
               --lat 62.17 --lon 47.64 --alt 582865 \
               --head 328.8 --tilt 68.9 --roll 177.6
```
