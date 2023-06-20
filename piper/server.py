import asyncio
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol, serve
from typing import Any
from loadbalancer import LoadBalancer


class Server:
    def __init__(self, load_balancer: LoadBalancer) -> None:
        self.load_balancer = load_balancer

    async def handle_connection(
        self, websocket: WebSocketServerProtocol, _path: str
    ) -> None:
        try:
            while True:
                # Receive image frames from the WebSocket connection
                data = await websocket.recv()

                # Process the received data (e.g., convert it to a task)
                task = self.process_received_data(data)

                # Put the task into the input pipe of the load balancer for processing
                self.load_balancer.input_pipe.send(task)
        except ConnectionClosed:
            # TODO: Discard all of the worker shit here
            pass

    def process_received_data(self, data: str) -> Any:
        # Process input data here to np.array and feed into workers etc
        return data

    def start_websocket_server(self) -> None:
        # WebSocket server configuration
        host = "0.0.0.0"
        port = 8000

        # Start the WebSocket server
        start_server = ws.serve(
            lambda websocket, path: self.handle_connection(websocket, path), host, port
        )

        # Event loop for the WebSocket server
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
