import logging as log

from cv_entrypoint import open_cv_frame_piper

if __name__ == "__main__":
    log.info("PiperWorker starting...")
    open_cv_frame_piper()
