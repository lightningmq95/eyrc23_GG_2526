import socket
from time import sleep
import signal
import sys

def signal_handler(sig, frame):
    print('Clean-up!')
    cleanup()
    sys.exit(0)

def cleanup():
    if s.fileno() != -1:
        s.close()
        print("Cleanup done")
    else:
        print("Socket already closed") 

ip = "192.168.29.68"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect((ip, 8002))
    
    print("Connected to ESP32")
    while True:
        # Get user input (1 to move, 0 to stop)
        user_input = input("Enter command (1 to move, 0 to stop): ")

        # Send command to motor
        s.sendall(str.encode(user_input))