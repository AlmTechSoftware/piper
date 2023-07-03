# main.py
import os
import sys
import signal
import asyncio
import logging
import http.server
import socketserver
import websockets
import uuid

from typing import List
from http import HTTPStatus
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", 4242))
MAX_WORKER_COUNT = int(os.getenv("MAX_WORKER_COUNT", 0))
BUFFER_SIZE = 1024

workers: List['PiperWorker'] = []


async def handle_client(websocket):
    conn = PiperConnection(websocket)

    await conn.send_message("Connected to the Piper Server!")

    worker = PiperWorker()
    workers.append(worker)

    while True:
        try:
            frame_data = await conn.receive_message()

            if not frame_data:
                break

            processed_data = await worker.process_frame(frame_data, BUFFER_SIZE)

            if processed_data:
                await conn.send_message(processed_data)
        except Exception as e:
            logging.error(f"Error occurred while handling client: {e}")
            break

    workers.remove(worker)


class PiperConnection:
    def __init__(self, websocket: WebSocketServerProtocol):
        self.websocket = websocket

    async def send_message(self, message: str):
        await self.websocket.send(message)

    async def receive_message(self) -> bytes:
        return await self.websocket.recv()


class PiperWorker:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.socket = f"/tmp/piperworker_{self.id}.socket"
        self.ipc = ipc.Client(self.socket)
        self.process = None

    async def process_frame(self, frame_data: bytes, buffer_size: int) -> bytes:
        await self.send(frame_data)

        try:
            processed_data = await self.recv(buffer_size)
            return processed_data
        except Exception as e:
            logging.error(f"Frame data processing failed: {e}")

        return b""

    async def send(self, data: bytes):
        await self.ipc.send(ipc.Message(data=data))

    async def recv(self, size: int) -> bytes:
        message = await self.ipc.receive()
        return message.data

    def kill(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

        self.ipc.close()


def init_workers(num_workers: int) -> List[PiperWorker]:
    workers = []
    for _ in range(num_workers):
        worker = PiperWorker()
        workers.append(worker)
    return workers


def signal_handler(sig, frame):
    for worker in workers:
        worker.kill()

    sys.exit(0)


def main():
    num_workers = int(os.getenv("MAX_WORKER_COUNT", 0))

    workers.extend(init_workers(num_workers))

    logging.basicConfig(level=logging.INFO)

    server = socketserver.TCPServer(("0.0.0.0", PORT), http.server.SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    signal.signal(signal.SIGINT, signal_handler)

    start_server = websockets.serve(handle_client, "0.0.0.0", PORT)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
