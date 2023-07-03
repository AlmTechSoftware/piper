import asyncio
import numpy as np
import cv2
import websockets
import multiprocessing

async def video_processing(frame):
    # Perform video processing using NumPy or any other desired method
    # This is just a placeholder example
    processed_frame = np.flip(frame, axis=1)  # Flip the frame horizontally
    return processed_frame

async def handle_websocket(websocket, path):
    while True:
        try:
            data = await websocket.recv()
            # Assuming the received data is the raw H.264 video frame
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # Create a new process to handle video processing
            process = multiprocessing.Process(target=video_processing, args=(frame,))
            process.start()

        except websockets.exceptions.ConnectionClosed:
            break

async def start_server():
    websocket_server = websockets.serve(handle_websocket, 'localhost', 8765)

    async with websocket_server:
        await websocket_server.start_serving()

def run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_server())
    loop.run_forever()

if __name__ == '__main__':
    run_server()
