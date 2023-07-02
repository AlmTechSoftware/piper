import os
import cv2
import sys
import numpy as np
import logging as log
import asyncio
# from cv_entrypoint import open_cv_frame_piper


def process_frame(frame_data_bytes):
    # TODO: Process the received frame data

    # Convert the frame data to a NumPy array
    frame_array = np.frombuffer(frame_data_bytes, dtype=np.uint8)

    # Decode the H.264 frame using OpenCV
    new_frame = cv2.imdecode(frame_array, cv2.IMREAD_UNCHANGED)

    # Send back the frame in byte format
    return new_frame.tobytes()


async def handle_client(reader, writer):
    client_address = writer.get_extra_info("peername")
    print(f"Accepted connection from {client_address}")

    try:
        while True:
            # Read data from the client
            data = await reader.read(1024)
            if not data:
                break

            # Process data
            new_frame = process_frame(data)

            # Send back the processed data
            writer.write(new_frame)
            await writer.drain()

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Close the client connection
        writer.close()
        print(f"Closed connection from {client_address}")


async def start_worker(socket_path: str):
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
    print(f"Starting PiperWorker on socket \"{socket_path}\"")

    asyncio.run(start_worker(socket_path))
