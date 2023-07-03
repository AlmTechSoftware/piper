import os
import logging
from dotenv import load_dotenv
from .server import run_server
from colored import Fore, Back, Style

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format=f"{Fore.rgb(100, 100, 100)}%(asctime)s {Fore.rgb(255, 255, 255)}{Back.rgb(80, 200, 80)} PIPER {Style.reset} [{Fore.rgb(255, 240, 240)}%(levelname)s{Style.reset}] %(message)s",
)

PORT = int(os.getenv("PORT", 4242))


def main():
    run_server("/piper", PORT)


if __name__ == "__main__":
    main()
