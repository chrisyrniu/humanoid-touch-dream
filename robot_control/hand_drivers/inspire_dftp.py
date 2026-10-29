"""
Double Inspire Hand Driver

This script runs two processes in parallel to control left and right Inspire hands
via Modbus communication. Each process continuously reads data from its respective hand.
"""

import multiprocessing
import time
from inspire_sdkpy import inspire_sdk


def worker(ip, lr, name, network=None):
    """Worker function that continuously reads data from an Inspire hand.
    
    Args:
        ip: IP address of the Inspire hand
        lr: Hand side ('l' for left, 'r' for right)
        name: Process name for logging
        network: Optional network configuration
    """
    handler = inspire_sdk.ModbusDataHandler(network=network, ip=ip, LR=lr, device_id=1)

    call_count = 0
    start_time = time.perf_counter()
    time.sleep(0.5)  # Initial delay for connection setup
    
    try:
        while True:
            data_dict = handler.read()
            call_count += 1
            time.sleep(0.001)
            
            # Print statistics every 10 calls
            if call_count % 10 == 0:
                elapsed_time = time.perf_counter() - start_time
                frequency = call_count / elapsed_time
                print(f"{name} - Current frequency: {frequency:.2f} Hz, "
                      f"Call count: {call_count}, Elapsed time: {elapsed_time:.6f} s")
    except KeyboardInterrupt:
        elapsed_time = time.perf_counter() - start_time
        frequency = call_count / elapsed_time if elapsed_time > 0 else 0
        print(f"{name} - Process ended. Total calls: {call_count}, "
              f"Total time: {elapsed_time:.6f} s, Final frequency: {frequency:.2f} Hz")


if __name__ == "__main__":
    # Create processes for right and left hands with their respective IP addresses
    process_r = multiprocessing.Process(
        target=worker, 
        args=('192.168.123.211', 'r', "Right Hand Process")
    )
    process_l = multiprocessing.Process(
        target=worker, 
        args=('192.168.123.210', 'l', "Left Hand Process")
    )

    # Start processes with a slight delay
    process_r.start()
    time.sleep(0.6)
    process_l.start()

    try:
        # Keep main process alive
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        # Clean shutdown
        process_r.terminate()
        process_l.terminate()
