from .node import section_frames


def entrypoint(frame, contours):
    """
    The convex hull and canvas cleanup process.
    """

    mask_frame = section_frames(frame, contours)

    return frame, contours, mask_frame
