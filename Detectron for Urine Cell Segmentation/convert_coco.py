import os
import cv2
import numpy as np
from skimage.measure import label
from pycocotools.mask import encode
from detectron2.structures import BoxMode
import json

def get_cell_dicts(DIR):

    pixel = {"cytoplasm" : np.array([0, 255, 0]), "nucleus" : np.array([255, 0, 0])}

    dataset_dicts = []


    def getComponents(mask):
        labels, num = label(mask, return_num=True)
        object_masks = []
        for i in range(1, num+1):
            object_masks.append(np.array((labels == i), dtype=np.uint8))
        return object_masks


    for filename in os.listdir(DIR + "masks/"):
        if filename.endswith(".png"):
            instance = {}
            mask = cv2.imread(DIR + "masks/" + filename)

            instance["file_name"] =DIR + "imgs/" + filename
            instance["height"], instance["width"], x = mask.shape
            instance["image_id"] = filename.split(".")[0]

            annos = []
            for ID, cls in enumerate(pixel):
                bn_mask = np.array((mask == pixel[cls]).all(axis=2), dtype=int)
                object_masks = getComponents(bn_mask)

                for object_mask in object_masks:
                    annotation = {}
                    rle = encode(np.asarray(object_mask, order="F"))
                    points = np.argwhere(object_mask > 0)
                    x, y = points[:, 0], points[:, 1]

                    annotation["bbox"] = [float(min(x)), float(min(y)), float(max(x)), float(max(y))]
                    annotation["bbox_mode"] = BoxMode.XYXY_ABS
                    annotation["category_id"] = ID
                    annotation["segmentation"] = rle
                    annos.append(annotation)
            instance["annotations"] = annos
            dataset_dicts.append(instance)
    return dataset_dicts
