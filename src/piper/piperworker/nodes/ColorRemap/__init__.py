from .node import canvas_render


def entrypoint(frame, contours, mask_frame):
    """
    Entrypoint for color remap
    """
    return canvas_render(frame, contours, mask_frame)
