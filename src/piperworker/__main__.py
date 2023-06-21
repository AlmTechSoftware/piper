from os import environ
import logging as log

from cv_entrypoint import open_cv_frame_piper

if __name__ == "__main__":
    worker_name = environ.get("PIPERWORKER_NAME")
    log.info(f"Starting PiperWorker: {worker_name}")

    # Build the full path to the named pipe
    pipe_path = "/tmp/{worker_name}"

    # Start the worker process
    open_cv_frame_piper(pipe_path)
