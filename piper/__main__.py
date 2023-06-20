#!/usr/bin/env python

import os
import logging as log

# Get environment variables
IP = os.getenv("PIPER_IP") or "0.0.0.0"
PORT = int(os.getenv("PIPER_PORT") or 4489)
NUM_WORKERS = int(os.getenv("PIPER_WORKERS") or 8)


if __name__ == "__main__":
    log.info(f"Starting Piper server with {IP=} {PORT=} {NUM_WORKERS=}")
