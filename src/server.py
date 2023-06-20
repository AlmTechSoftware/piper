import asyncio
import websockets as ws


async def handle_connection(websocket, path, input_queue):
    try:
        while True:
            # Receive image frames from the WebSocket connection
            data = await websocket.recv()

            # Process the received data (e.g., convert it to a task)
            task = process_received_data(data)

            # Put the task into the input queue for processing
            input_queue.put(task)
    except ws.exceptions.ConnectionClosed:
        pass


def process_received_data(data):
    # Convert the data to the input type of the starting pipeline node
    return data


def start_websocket_server(input_queue, num_workers, output_pipes):
    # WebSocket server configuration
    host = "0.0.0.0"  # Set the host IP address
    port = 8000  # Set the desired port number

    # Start the WebSocket server
    start_server = ws.serve(
        lambda websocket, path: handle_connection(websocket, path, input_queue),
        host,
        port,
    )

    # Event loop for the WebSocket server
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
