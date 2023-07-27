import torch
import os
import cv2
import random
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset_handler import *

CHECKPOINT_PATH = "checkpoints/sam_vit_h.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

DATASET_LOCATION = "dataset/"
DATA_SET_SUBDIRECTORY = "train/"
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
IMAGES_DIRECTORY_PATH = os.path.join(DATASET_LOCATION, DATA_SET_SUBDIRECTORY)
ANNOTATIONS_FILE_PATH = os.path.join(
    DATASET_LOCATION, DATA_SET_SUBDIRECTORY, ANNOTATIONS_FILE_NAME
)

coco_data = load_coco_json(json_file=ANNOTATIONS_FILE_PATH)

CLASSES = [
    category.name
    for category in coco_data.categories
    if category.supercategory != "none"
]

IMAGES = [image.file_name for image in coco_data.images]
random.seed(10)

EXAMPLE_IMAGE_NAME = random.choice(IMAGES)
EXAMPLE_IMAGE_PATH = os.path.join(
    DATASET_LOCATION, DATA_SET_SUBDIRECTORY, EXAMPLE_IMAGE_NAME
)

# load dataset annotations
annotations = COCOJsonUtility.get_annotations_by_image_path(
    coco_data=coco_data, image_path=EXAMPLE_IMAGE_NAME
)
ground_truth = COCOJsonUtility.annotations2detections(annotations=annotations)

# small hack - coco numerate classes from 1, model from 0 + we drop first redundant class from coco json
ground_truth.class_id = ground_truth.class_id - 1

# load image
image_bgr = cv2.imread(EXAMPLE_IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# initiate annotator
box_annotator = sv.BoxAnnotator(color=sv.Color.red())
mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

# annotate ground truth
annotated_frame_ground_truth = box_annotator.annotate(
    scene=image_bgr.copy(), detections=ground_truth, skip_label=True
)

# run SAM inference
mask_predictor.set_image(image_rgb)

masks, scores, logits = mask_predictor.predict(
    box=ground_truth.xyxy[0], multimask_output=True
)

detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
detections = detections[detections.area == np.max(detections.area)]

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[annotated_frame_ground_truth, annotated_image],
    grid_size=(1, 2),
    titles=["source image", "segmented image"],
)
