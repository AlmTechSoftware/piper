import cv2
import numpy as np


def canvas_render(frame, contours, mask_frame):
    """
    This function takes the last bit of information and renderes
    the new canvas board.
    """

    colored_frame = np.zeros(frame.shape, np.uint8)

    frame = cv2.bitwise_and(frame, frame, mask=mask_frame)

    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    for cnt in contours:  # colors!!
        colors = []
        for px in cnt[::3]:  # Only every third, for efficiency
            col = frame[(px[0][1]), (px[0][0])]

            col = np.clip(1.2 * col + 20, 0, 255)

            colors.append((int(col[0]), int(col[1]), int(col[2])))

        avg_color = np.average(colors, axis=0)

        cv2.drawContours(colored_frame, [cnt], 0, avg_color, 1)

    return colored_frame
