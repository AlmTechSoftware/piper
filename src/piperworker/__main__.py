import os
import cv2
import sys
import numpy as np
import logging as log
import asyncio
from asyncio.streams import StreamReader, StreamWriter

socket_path: str = "/tmp/my_socket.socket"
max_queue_size: int = 10


async def process_frame(frame_data_bytes: bytes) -> bytes:
    # Convert the frame data to a NumPy array
    frame_array = np.frombuffer(frame_data_bytes, dtype=np.uint8)

    # Decode the H.264 frame using OpenCV
    new_frame = cv2.imdecode(frame_array, cv2.IMREAD_UNCHANGED)

    # TODO: Perform video processing 

    # Send back the frame in byte format
    return new_frame.tobytes()


async def handle_client(reader: StreamReader, writer: StreamWriter) -> None:
    client_address = writer.get_extra_info("peername")
    print(f"Accepted connection from {client_address}")

    # Create an asyncio Queue to store the video frames
    video_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queue_size)

    # Start the video processing task
    processing_task: asyncio.Task[None] = asyncio.create_task(
        process_video_frames(video_queue, writer)
    )

    try:
        while True:
            # Read data from the client
            data: bytes = await reader.read(1024)
            if not data:
                break

            # Put the received frame into the video queue
            try:
                await video_queue.put(data)
            except asyncio.QueueFull:
                print(f"Video queue is full, dropping frame...")

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Stop the video processing task
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass

        # Close the client connection
        writer.close()
        print(f"Closed connection from {client_address}")


async def process_video_frames(
    video_queue: asyncio.Queue[bytes], writer: StreamWriter
) -> None:
    while True:
        # Get the next frame from the video queue
        frame_data: bytes = await video_queue.get()

        # Process the frame asynchronously
        processed_frame: bytes = await process_frame(frame_data)

        # Send back the processed frame to the client
        writer.write(processed_frame)
        await writer.drain()

        # Example: Print the size of the processed frame
        print(f"Processed frame: {len(processed_frame)} bytes")


async def start_worker(socket_path: str) -> None:
    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = await asyncio.start_unix_server(handle_client, path=socket_path)

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.error(
            "Socket path not provided! Please do `python -m piperworker (socket path)`"
        )
        exit(1)

    socket_path = sys.argv[1]
    print(f'Starting PiperWorker on socket "{socket_path}"')

    if os.path.exists(socket_path):
        os.remove(socket_path)

    asyncio.run(start_worker(socket_path))
