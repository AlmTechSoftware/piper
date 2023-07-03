import multiprocessing
import numpy as np
import asyncio
import logging
import cv2


async def piper_entrypoint(frame):
    # TODO: Run the piper entrypoint here and return the new frame

    return frame


async def parse_client_data(data: bytes, client_ip_str: str = ""):
    try:
        # Assuming the received data is the raw H.264 video frame
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as err:
        logging.error(
            f"{client_ip_str}: Error parsing data sent by client {err=} {data=}"
        )
        return None

    return frame


def process_frame(frame, queue):
    processed_frame = asyncio.run(piper_entrypoint(frame))
    queue.put(processed_frame)


async def process(data, loop, client_ip_str: str = ""):
    frame = await parse_client_data(data, client_ip_str)

    if frame:
        # Create a new process to handle video processing
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=process_frame, args=(frame, queue))
        process.start()

        # Wait for the processing to finish
        processed_frame = await loop.run_in_executor(None, queue.get)

        return processed_frame
    else:
        return None
