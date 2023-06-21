from multiprocessing import Process, Pipe


def frame_processing(child_conn):
    # Frame processing code
    while True:
        frame = child_conn.recv()  # Receive a frame from the parent process
        if frame is None:
            continue

        # Convert the frame to grayscale
        # grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Send frame to the algorithm
        new_frame = None # TODO: 

        # Send the processed frame back to the parent process
        child_conn.send(new_frame)

    # child_conn.close()


def open_cv_frame_piper():
    parent_conn, child_conn = Pipe()

    # Start the child process for frame processing
    p = Process(target=frame_processing, args=(child_conn,))
    p.start()

    # Main loop for reading frames and sending them to the child process
    while True:
        frame = None  # TODO: impl frame reading logic
        if frame is None:
            break

        # Send the frame to the child process for processing
        parent_conn.send(frame)

        # Receive the processed frame from the child process
        new_frame = parent_conn.recv()

        # Use the grayscale frame for further processing or streaming

    # Signal the child process to exit
    parent_conn.send(None)

    # Wait for the child process to finish
    p.join()
