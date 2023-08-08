import socket

def send_udp_message(message, address='localhost', port=4080):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode(), (address, port))
    finally:
        sock.close()

send_udp_message('2023-07-03-23-34-53,3142dea,exit')