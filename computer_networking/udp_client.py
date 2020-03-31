from socket import *

server_name = '192.168.50.65'
server_port = 12000
# use ipv4 and UDP protocol
client_socket = socket(AF_INET, SOCK_DGRAM)
message = raw_input("input lowercase sentence:")
client_socket.sendto(message.encode(), (server_name, server_port))
modified_message, server_address = client_socket.recvfrom(2048)

print(modified_message.decode())
client_socket.close()


