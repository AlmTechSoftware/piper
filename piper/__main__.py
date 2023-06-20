#!/usr/bin/env python

from multiprocessing import Queue
import os
import logging as log

from piper.loadbalancer import LoadBalancer
from piper.server import Server

# Get environment variables
HOST = os.getenv("PIPER_IP") or "0.0.0.0"
PORT = int(os.getenv("PIPER_PORT") or 4489)
NUM_WORKERS = int(os.getenv("PIPER_WORKERS") or 8)

if __name__ == "__main__":
    log.info(f"Starting Piper server with {HOST=} {PORT=} {NUM_WORKERS=}")

    # Create the input pipe
    input_pipe = Queue()

    # Initialize the loadbalancer
    load_balancer = LoadBalancer(input_pipe=input_pipe, num_workers=NUM_WORKERS)

    # Create the server and start listening
    server = Server(host=HOST, port=PORT, load_balancer=load_balancer)
    server.start_websocket_server()
