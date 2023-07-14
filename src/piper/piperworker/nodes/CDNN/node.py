import numpy as np
import cv2 as cv
from ultralytics import YOLO

ss_net = YOLO("./model/yolov8n-seg.pt")


def find_mask_frame(frame):
    """
    The function for finding items and returning their mask
    """

    global ss_net
    width, height, _ = frame.shape  # Get dimensions for the current frame

    result = ss_net(frame)  # Put frame into the net
    res_boxes = result[0].boxes  # Get bounding boxes
    res_masks = result[0].masks  # Get masks

    if (
        res_masks == None
    ):  # Error handling, if nothing is found, just return an empty frame
        return np.zeros((width, height))

    classes = np.array(res_boxes.cls.cpu(), dtype="int")
    masks = res_masks.data.cpu().numpy()

    mask_frame = np.zeros(
        (width, height)
    )  # Currently an empty frame, but will be modified to house the mask (mainly for visualization)

    for clss, mask in zip(classes, masks):
        if clss != 60:  # Class 60 is a table, just testing stuff
            continue

        mask = mask * 255  # Since mask is just ones and zeroes
        mask = cv.resize(mask, (height, width))

        mask_frame[
            0:width, 0:height
        ] = mask  # Draw the contours on the empty mask_frame

    return mask_frame


def get_contour_corners(mask_frame):
    """
    Takes in a mask and returns the corner points
    """

    mask_frame = np.uint8(mask_frame)

    contours, _hierarchy = cv.findContours(
        mask_frame, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
    )

    for c in contours:
        epsilon = 0.05 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)

        points = [(a[0][0], a[0][1]) for a in approx]

    return points
