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
