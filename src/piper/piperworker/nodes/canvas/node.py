import numpy as np
import cv2
from typing import Any, List


def section_frames(frame: Any, contours: List[Any]) -> Any:
    """
    A function that takes the frame along with its contours to create a new mask
    This new mask contains the bounding boxes of the contours
    """
    mask_frame = np.zeros(frame.shape[:2], np.uint8)

    contours_poly = [None] * len(contours)
    bound_box = [None] * len(contours)

    for i, cont in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(cont, 3, True)
        bound_box[i] = cv2.boundingRect(contours_poly[i])

    # Note: some smaller bboxes become subsets of the larger ones due to the for-loop solution
    for i in range(len(contours)):
        cv2.rectangle(
            mask_frame,
            (int(bound_box[i][0]), int(bound_box[i][1])),
            (
                int(bound_box[i][0] + bound_box[i][2]),
                int(bound_box[i][1] + bound_box[i][3]),
            ),
            (255, 255, 255),
            cv2.FILLED,
        )

    return mask_frame
