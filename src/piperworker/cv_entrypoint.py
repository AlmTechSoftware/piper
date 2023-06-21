import os
import numpy as np
from multiprocessing import Process, Pipe


def frame_processing(child_conn):
    # Frame processing code
    while True:
        frame = child_conn.recv()  # Receive a frame from the parent process
        if frame is None:
            break

        # Send frame to the algorithm
        new_frame = None  # TODO: Process the frame using your algorithm

        # Send the processed frame back to the parent process
        child_conn.send(new_frame)

    child_conn.close()


def open_cv_frame_piper(pipe_path):
    parent_conn, child_conn = Pipe()

    # Start the child process for frame processing
    p = Process(target=frame_processing, args=(child_conn,))
    p.start()

    # Open the named pipe for reading frames from the Golang program
    pipe_fd = os.open(pipe_path, os.O_RDONLY)

    # Main loop for reading frames and sending them to the child process
    while True:
        frame_data = os.read(pipe_fd, 65536)  # Read frame data from the named pipe
        if not frame_data:
            continue

        # Process the frame data if necessary (e.g., decode, convert to numpy array)

        # INFO: apt install libswscale-dev libavcodec-dev libavutil-dev
        framedatas = decoder.decode(frame_data)
        for framedata in framedatas:
            (frame, w, h, ls) = framedata
            if frame is not None:
                # print('frame size %i bytes, w %i, h %i, linesize %i' % (len(frame), w, h, ls))
                frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
                frame = frame.reshape((h, ls // 3, 3))
                frame = frame[:, :w, :]

                # Send the frame to the child process for processing
                parent_conn.send(frame_data)

        # Receive the processed frame from the child process
        new_frame = parent_conn.recv()

    # Signal the child process to exit
    parent_conn.send(None)

    # Wait for the child process to finish
    p.join()

    # Close the named pipe
    os.close(pipe_fd)
    os.remove(pipe_path)
