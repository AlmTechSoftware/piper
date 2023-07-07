import multiprocessing
import numpy as np
import h264decoder
import asyncio
import logging
import cv2

from piper.piperworker.binder import get_pipeline

decoder = h264decoder.H264Decoder()


async def piper_entrypoint(frame):
    pipeline = get_pipeline()

    return pipeline.eval(frame)


async def parse_client_data(data: bytes, client_ip_str: str = ""):
    try:
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return frame
        # H.264 video frame decoding
        # framedatas = decoder.decode(data)
        # for framedata in framedatas:
        #     (frame, w, h, ls) = framedata
        #     if frame is not None:
        #         frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
        #         frame = frame.reshape((h, ls // 3, 3))
        #         frame = frame[:, :w, :]
        #
        #         return frame
    except Exception as err:
        logging.error(
            f"{client_ip_str}: Error parsing data sent by client {err=} {data=}"
        )
        return None


def process_frame(frame, queue):
    processed_frame = asyncio.run(piper_entrypoint(frame))
    queue.put(processed_frame)


async def process(data, loop, client_ip_str: str = ""):
    frame = await parse_client_data(data, client_ip_str)

    if frame is not None and frame.any():
        # Create a new process to handle video processing
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=process_frame, args=(frame, queue))
        process.start()

        # Wait for the processing to finish
        processed_frame = await loop.run_in_executor(None, queue.get)

        return processed_frame
    else:
        return None
