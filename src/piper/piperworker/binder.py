from .pipeline import Pipeline

# Pseudo code, but the basis
# import canvas
# import CDNN
# import CEMNN
# import ColorRemap
# import CRAD
# import SGNN
# import vect

global pipeline
pipeline = Pipeline()


def get_pipeline():
    global pipeline
    return pipeline


# Mockup for pipeline
# pipeline.bind(canvas.entrypoint())
# pipeline.bind(CDNN.entrypoint())
# pipeline.bind(CEMNN.entrypoint())
# pipeline.bind(ColorRemap.entrypoint())
# pipeline.bind(CRAD.entrypoint())
# pipeline.bind(SGNN.entrypoint())
# pipeline.bind(vect.entrypoint())
