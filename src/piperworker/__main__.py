import cv2
import sys
import socket
import numpy as np
import logging as log
# from cv_entrypoint import open_cv_frame_piper


def process_frame(frame_data_bytes):
    # TODO: Process the received frame data

    # Convert the frame data to a NumPy array
    frame_array = np.frombuffer(frame_data_bytes, dtype=np.uint8)

    # Decode the H.264 frame using OpenCV
    new_frame = cv2.imdecode(frame_array, cv2.IMREAD_UNCHANGED)

    # Send back the frame in byte format
    return new_frame.tobytes()


def start_worker(socket_path: str):
    # Create a Unix domain socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    try:
        # Connect to the socket
        sock.connect(socket_path)
        print("Connected to the socket:", socket_path)

        # Process frame data received from the Go server
        while True:
            frame_data = sock.recv(1024)
            if not frame_data:
                continue

            # Process the frame data
            processed_data = process_frame(frame_data)

            # Send the processed data back to the Go server
            sock.sendall(processed_data)
    except Exception as err:
        log.error("Error occurred:", err)
    finally:
        # Close the socket
        sock.close()
        print("Socket connection closed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.error(
            "Socket path not provided! Please do `python -m piperworker (socket path)`"
        )
        exit(1)

    socket_path = sys.argv[1]
    print(f'Starting PiperWorker on socket "{socket_path}"')

    start_worker(socket_path)

    # Start the worker process
    # open_cv_frame_piper(socket)
