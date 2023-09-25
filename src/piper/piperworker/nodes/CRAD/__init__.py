from .node import remove_perspective, canvas_contour


def entrypoint(frame, points):
    """
    The finalization of the entire CRAD process.
    """

    crad_frame = remove_perspective(frame, points)
    crad_frame, frame_contours = canvas_contour(crad_frame)

    return crad_frame, frame_contours
