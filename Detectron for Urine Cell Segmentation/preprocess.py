import sys
import os
import cv2

SOURCE_DIR = sys.argv[1] #"PreliminaryGenerativeHistoPath/cyto2label_public/train/"
SAVE_DIR = sys.argv[2]

for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".png"):
        path = SOURCE_DIR + filename
        img = cv2.imread(path)
        l = int(img.shape[1]/2)
        img, mask = img[:, :l], img[:, l:]
        cv2.imwrite(SAVE_DIR + "imgs/" + filename, img)
        cv2.imwrite(SAVE_DIR + "masks/" + filename, mask)





