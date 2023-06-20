import asyncio
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol, serve
from typing import Any
from .loadbalancer import LoadBalancer


class Server:
    def __init__(
        self, load_balancer: LoadBalancer, host: str = "0.0.0.0", port: int = 4489
    ) -> None:
        self.load_balancer = load_balancer
        self.host = host
        self.port = port

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
        # TODO: Process input data here to np.array and feed into workers etc
        return data

    def start_websocket_server(self) -> None:
        # Start the WebSocket server
        start_server = serve(
            lambda websocket, path: self.handle_connection(websocket, path),
            self.host,
            self.port,
        )

        # Event loop for the WebSocket server
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
