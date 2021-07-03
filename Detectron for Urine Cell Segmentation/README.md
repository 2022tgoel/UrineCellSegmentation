# Detectron for Urine Cell Segmentation

# Scripts
* `preprocess.py` 
    * splits up the segmentation and mask portion
* `convert_coco.py`
    * contains the utility function for creating a JSON in the COCO data format
    * Resources for understanding COCO
        * https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
        * https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
    * Methods
        1. Identify all connected components of the same color
            * findComponents(class): returns a mask for each connected region of given color (the class)
            * https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
        2. Convert that to run length encodings
            * `pycocotools.mask.encode(np.asarray(mask, order="F"))`
* `training.py`
    * trains the model
* `gen_output.py`
    * creates some images with the segmentation visualization overlayed
     

# Other Resources
[Data Source](https://github.com/jlevy44/PreliminaryGenerativeHistoPath/)
[Detectron Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=U5LhISJqWXgM)

# TODO

- [ ] check the output to make sure it's sensible (https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format)
- [ ] fix the hacky cfg.INPUT.MASK_FORMAT='bitmask' thing I did (probably should just create a polygon as opposed to the RLE)
- [ ] check skip loading parameter warnings 
- [x] put notebook code in scripts
- [ ] pickle the json
- [ ] still need to figure out why the +0.5 ... 
- [ ] doesn't seem like the model converged - train longer