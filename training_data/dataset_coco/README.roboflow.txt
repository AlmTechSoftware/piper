
FeynMAN - v7 2023-08-20 12:29am
==============================

This dataset was exported via roboflow.com on August 19, 2023 at 10:29 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 80 images.
Sghug are annotated in COCO Segmentation format.

The following pre-processing was applied to each image:

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -45 and +45 degrees
* Random shear of between -45° to +45° horizontally and -45° to +45° vertically

The following transformations were applied to the bounding boxes of each image:
* Random exposure adjustment of between -25 and +25 percent
* Salt and pepper noise was applied to 5 percent of pixels


