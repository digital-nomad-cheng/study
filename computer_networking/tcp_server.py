from socket import *

server_port = 12000
server_socket = socket(AF_INET, SOCK_STREAM)
server_socket.bind(('', server_port))
server_socket.listen(1)
print('server is ready to receivce')
while True:
    connetcion_socket, addr = server_socket.accept()
    message, client_address = connetion_socket.recv(2048)
    modified_message = message.decode().upper()
    connetion_socket.send(modified_message.encode())
    connection_socket.close()

