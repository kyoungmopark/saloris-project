import socket

def start_udp_server(ip: str, port: int):
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind((ip, port))

    print(f"UDP Server started on {ip}:{port}")
    
    try:
        while True:
            data, addr = server.recvfrom(65535)  # buffer size is 65535 bytes
            print(f"Received message: {data.decode('utf-8')} from {addr}")
    except KeyboardInterrupt:
        print("\nUDP Server shutting down...")
    finally:
        server.close()

if __name__ == "__main__":
    start_udp_server("127.0.0.1", 4455)