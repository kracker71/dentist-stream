import os
import sys
import logging

import grpc
import socket
import asyncio
import contextlib

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from pbs import speech2text_pb2_grpc
from speech2text_grpc_servicer import GowajeeSpeechRecognizerService

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 1))
LOG_FORMAT = "{levelname} [{filename}:{lineno}]:"
LOG_LEVEL: str = "INFO"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def _run_server(bind_address):
    logger.debug(f"Server started. Awaiting jobs...")
    server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=1),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            ("grpc.so_reuseport", 1),
            ("grpc.use_local_subchannel_pool", 1),
        ],
    )
    # image_ocr_pb2_grpc.add_OCRServicer_to_server(OCRService, server)
    speech2text_pb2_grpc.add_GowajeeSpeechToTextServicer_to_server(GowajeeSpeechRecognizerService(), server)
    server.add_insecure_port(bind_address)
    await server.start()
    await server.wait_for_termination()
    
def _run(bind_address):
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(_run_server(bind_address))

@contextlib.contextmanager
def _reserve_port():
    """Find and reserve a port for all subprocesses to use"""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("", 50051))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def main():
    """
    Inspired from https://github.com/grpc/grpc/blob/master/examples/python/multiprocessing/server.py
    """
    logger.info(f"Initializing server with {NUM_WORKERS} workers")
    with _reserve_port() as port:
        bind_address = f"[::]:{port}"
        logger.info(f"Binding to {bind_address}")
        sys.stdout.flush()
        workers = []
        for _ in range(NUM_WORKERS):
            worker = multiprocessing.Process(target=_run, args=(bind_address,))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()


if __name__ == "__main__":

    def serve(address: str) -> None:
        server = grpc.server(ThreadPoolExecutor(max_workers=4))
        speech2text_pb2_grpc.add_GowajeeSpeechToTextServicer_to_server(GowajeeSpeechRecognizerService(), server)
        server.add_insecure_port(address)
        server.start()
        logging.info("Server serving at %s", address)
        server.wait_for_termination()

    serve('[::]:50051')
    