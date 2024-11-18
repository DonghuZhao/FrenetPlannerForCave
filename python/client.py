import socket
import json


def udp_client(server_host='127.0.0.1', server_port=6005):
    # 创建UDP套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # json_str = "S"
    # # 发送数据
    # client_socket.sendto(json_str.encode(), (server_host, server_port))
    try:
        send_data =     {
        "duration": 16.4978,
        "currentframeid": 50,
        "V": [{
            "id": "e_0",
            "X": 337738,
            "Y": 406784,
            "yaw": 46.4,
            "Vx": 500,
            "Vy": 1000,
            "ax": 0,
            "ay": 0
        }, {
            "id": "p_0",
            "X": 335269,
            "Y": 404605,
            "yaw": -46.4,
            "Vx": 0,
            "Vy": 0,
            "ax": 0,
            "ay": 0
        }]
    }
        json_str = json.dumps(send_data)
        # 发送数据
        client_socket.sendto(json_str.encode(), (server_host, server_port))
        print(f"发送数据成功")

        json_str = "E"
        # 发送数据
        client_socket.sendto(json_str.encode(), (server_host, server_port))
    finally:
        pass

if __name__ == '__main__':
    udp_client()

