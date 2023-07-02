import numpy as np
import cv2 as cv
from ultralytics import YOLO

ss_net = YOLO("./models/yolov8n-seg.pt")


def canvas_detection(frame):
    global ss_net
    print("Identifying people...")
    width, height, channels = frame.shape  # Get dimensions for the current frame

    result = ss_net(frame)  # Put frame into the net
    res_boxes = result[0].boxes  # Get bounding boxes
    res_masks = result[0].masks  # Get masks

    if res_masks == None:
        return frame

    bboxes = np.array(res_boxes.xyxy.cpu(), dtype="int")
    classes = np.array(res_boxes.cls.cpu(), dtype="int")
    masks = res_masks.data.cpu().numpy()

    mask_frame = np.zeros(
        (width, height, channels)
    )  # Currently an empty frame, but will be modified to house the mask (mainly for visualization)

    for bbox, clss, mask in zip(bboxes, classes, masks):
        if clss != 0:  # Class 0 is a person, and we are only interested in them
            continue

        (x, y, w, h) = bbox
        cv.rectangle(frame, (x, y), (w, h), (255, 0, 0))  # Draw bounding box

        mask = mask * 255
        mask = cv.resize(mask, (height, width))

        for channel in range(3):
            mask_frame[
                0:width, 0:height, channel
            ] = mask  # Draw mask (once for every channel)

    overlay = ((0.8 * frame) + (0.2 * mask_frame)).astype(
        "uint8"
    )  # Adds the input frame to the mask frame to create an overlay

    return overlay


def entrypoint(frame):
    cdnn_frame = frame

    return cdnn_frame
