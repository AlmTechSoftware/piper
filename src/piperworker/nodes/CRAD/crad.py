from typing import Any, List, Tuple
import cv2
import numpy as np


# Define the function to rotate an object to
# make it orthogonal to the screen
def remove_perspective(frame: Any, points: list) -> Tuple[int, int]:
    """
    Function to align the frame in 3D space
    given some normal.

    """

    def __perspective_dim(points: list):
        """
        Function to determine the output dimensions
        of the given points of a rectangle to be
        unperspectivized.
        """

        points = list(map(np.asarray, points))

        width_vec = max(
            points[1] - points[0],
            points[3] - points[2],
            key=np.linalg.norm,
        )

        height_vec = max(
            points[3] - points[0],
            points[2] - points[1],
            key=np.linalg.norm,
        )

        width, height = np.linalg.norm(width_vec), np.linalg.norm(height_vec)
        width, height = np.ceil(width), np.ceil(height)
        width, height = int(width), int(height)

        return width, height

    # Define the four corners of the desired rectangle
    perspective_src = np.array(points)

    # Calculate the width and height of the output
    width, height = __perspective_dim(points)

    # Define the corresponding four corners in the output image
    perspective_dist = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )

    # Calculate the perspective transform matrix
    perspective_transform, _ = cv2.findHomography(perspective_src, perspective_dist)

    # Apply the perspective transformation to the input image
    new_frame = cv2.warpPerspective(frame, perspective_transform, (width, height))

    return new_frame


def canvas_contour(
    canvas_frame: Any,
    board_color: Tuple[int, int, int] = (100, 100, 100),
    epsilon: int = 20,
) -> Tuple[Any, List[Any]]:
    """
    A function for getting the contours.
    """
    # Define the upper and lower bounds for the background color

    # Apply Gaussian blur to the result canvas_frame
    blur = cv2.GaussianBlur(canvas_frame, (5, 5), 0)

    # Convert the canvas_frame to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_RGBA2GRAY)

    # Apply edge detection to the blurred canvas_frame
    edges = cv2.Canny(gray, 10, 50)

    # Find contours in the edge map
    contours, _hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return canvas_frame, contours


def canvas_render(
    canvas_frame: Any,
    contours: List[Any],
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    pen_color: Tuple[int, int, int] = (255, 255, 255),
) -> Any:
    """
    This function takes the last bit of information and renderes
    the new canvas board.
    """

    # Make an empty frame
    rendered_frame = np.zeros(canvas_frame.shape[:3], np.uint8)

    # Apply the bg color
    rendered_frame[::] = bg_color

    # Approx contours
    for cnt in contours:
        cv2.drawContours(rendered_frame, [cnt], -1, pen_color, 1)

    return rendered_frame
