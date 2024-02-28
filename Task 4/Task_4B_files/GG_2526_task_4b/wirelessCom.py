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
        print("Socket is already closed")

ip = "192.168.137.240"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect((ip, 8002))
    
    print("Connected to ESP32")
    
    #activate Buzzer and led for 1 sec
    s.sendall(b'1')
    sleep(1)
    # Send command to start the motor (assuming 1 means start)
    s.sendall(b'2')
    
    while True:
       
        # Add signal handler to perform clean-up on Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Keep the program running
            while True:
                data = s.recv(1024)
                if not data:
                    break
                received_data = data.decode().strip()  # Remove leading/trailing whitespaces and newline characters
                print("Received: ", received_data)
                # sleep(1)  # Sleep for a second, you can adjust as needed
                if received_data == '11':
                    sleep(1.5)
                    s.sendall(b'0')
        except KeyboardInterrupt:
            break