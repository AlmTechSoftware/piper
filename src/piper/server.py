import asyncio
import websockets
import logging

from .process import process

from colored import Fore, Back, Style

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


async def handle_client(ws, path):
    if path != PATH:
        await ws.close()
        return

    client_ip = ws.remote_address[0]
    client_ip_str = f"{Fore.rgb(100, 200, 200)}{client_ip}{Style.reset}"
    logging.info(f"{client_ip_str}: connected")
    while True:
        try:
            data = await ws.recv()
            logging.info(f"{client_ip_str}: Received frame for processing")

            processed_frame = await process(data, loop, client_ip_str)

            if processed_frame is not None:
                logging.info(f"{client_ip_str}: Frame processed successfully")

                # Send the processed frame back to the client
                await ws.send(processed_frame.tobytes())

                logging.info(f"{client_ip_str}: New frame sent")
            else:
                logging.warn(f"{client_ip_str}: Malformed frame, skipping...")
                await ws.send(b"")
                continue

        except websockets.exceptions.ConnectionClosed:
            logging.info(f"{client_ip_str}: Disconnected.")
            break

        except asyncio.CancelledError:
            logging.info("Server is closing...")
            await ws.close()


async def start_server(path: str, port: int):
    global PATH
    PATH = path
    await websockets.serve(handle_client, "localhost", port)


def run_server(path: str = "/piper", port: int = 4242):
    loop.run_until_complete(start_server(path, port))
    loop.run_forever()
