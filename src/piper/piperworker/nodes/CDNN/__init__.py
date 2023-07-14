from .node import find_mask_frame
from .node import get_contour_corners


def entrypoint(frame):
    """
    The CDNN process, returns the original frame (unchanged) and the coordinates for the corners
    """
    cdnn_mask = find_mask_frame(frame)
    points = get_contour_corners(cdnn_mask)

    return frame, points
