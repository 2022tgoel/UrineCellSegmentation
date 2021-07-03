import sys
import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="directory containing the images")
parser.add_argument("--output_dir", help="directory to store output in")
args = parser.parse_args()

SOURCE_DIR = args.input_dir
if args.output_dir:
    SAVE_DIR = args.output_dir
else:
    SAVE_DIR = os.path.join("data/", os.path.basename(os.path.normpath(SOURCE_DIR)))

os.makedirs(os.path.join(SAVE_DIR, "imgs"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "masks"), exist_ok=True)

for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".png"):
        path = os.path.join(SOURCE_DIR, filename)
        img = cv2.imread(path)
        l = int(img.shape[1]/2)
        img, mask = img[:, :l], img[:, l:]
        cv2.imwrite(os.path.join(SAVE_DIR, "imgs", filename), img)
        cv2.imwrite(os.path.join(SAVE_DIR, "masks", filename), mask)





