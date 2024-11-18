import socket
import time
import pickle
from frenet_optimal_trajectory import frenet_optimal_planning, FrenetPath
import numpy as np
from Coordination import *
import json
import matplotlib.pyplot as plt

class TrajPlanQuest:
    def __init__(self):
        self.s = 0
        self.speed = 0
        self.acc = 0
        self.c_d = 0
        self.c_d_d = 0
        self.c_d_dd = 0
        self.ob = np.array([0,0])
        self.ob_v = np.array([0,0])

def handle_path(path):
    global csp
    path.x, path.y, path.yaw, path.Vx, path.Vy, path.ax, path.ay = ([0] * len(path.s) for _ in range(7))
    for i in range(len(path.s)):
        path.x[i], path.y[i], path.yaw[i], path.Vx[i], path.Vy[i], path.ax[i], path.ay[i] = (
            Frenet_to_Cartesian(csp,path.s[i],path.s_d[i],path.s_dd[i],path.d[i],path.d_d[i],path.d_dd[i]))
    return path

def data_input_handle(recv_data):
    '''
    :param recv_data:
    input
    {
        "duration": 16.4978,
        "currentframeid": 1,
        "V": [{
            "id": "e_0",			//车辆Id
            "X": 121.474000,		//X坐标
            "Y": 31.230001,		//Y坐标
            "yaw": 429,			//车辆转角
            "Vx": 0,				// X轴方向的速度
            "Vy": 0,				//Y轴方向的速度
            "ax": 0,				//X轴方向的加速度
            "ay": 0				//Y轴方向的加速度
        }, {
            "id": "p_0",			//行人Id
            "X": 121.474000,
            "Y": 31.230001,
            "yaw": 429,
            "Vx": 0,
            "Vy": 0,
            "ax": 0,
            "ay": 0
        }]
    }
    :return: quest -> TrajPlanQuest
    '''
    frame = recv_data["currentframeid"]
    quest = TrajPlanQuest()
    # 初始化
    ego_index = -1
    for index in range(len(recv_data["V"])):
        if recv_data["V"][index]["id"] == "controlled":
            ego_index = index
            break
    if ego_index == -1:
        return
    ego = recv_data["V"][ego_index]
    quest.s, quest.speed, quest.acc, quest.c_d, quest.c_d_d, quest.c_d_dd = (
        Cartesian_to_Frenet(ego["X"] / 100,ego["Y"] / 100,ego["Vx"] / 100,ego["Vy"] / 100,ego["ax"] / 100,ego["ay"] / 100, ego["yaw"] / 180 * 3.1415926))
    if len(recv_data["V"]) > 1:
        other_index = 0 if ego_index == 1 else 1
        other = recv_data["V"][other_index]
        quest.ob = np.array([other["X"] / 100,other["Y"] / 100])
        quest.ob_v = np.array([other["Vx"] / 100,other["Vy"] / 100])
    return [frame, quest]

def data_output_handle(frame, send_data):
    '''
    :param send_data: -> FrenetPath
    :return:
    {
        V:[
              {
                  'frameid':1
                  'info':[
                          {
                            "id": "e_0",
                            "X": 121.474000,
                            "Y": 31.230001,
                            "yaw": 429,
                            'Vx': 0,
                            'Vy': 0,
                            'ax': 0,
                            'ay': 0
                          }
                         ]
                }，
              {
                  ‘frameid':2
                  'info':[
                          {
                            "id": "e_0",
                            "X": 121.474000,
                            "Y": 31.230001,
                            "yaw": 429,
                            'Vx': 0,
                            'Vy': 0,
                            'ax': 0,
                            'ay': 0
                          }
                         ]
               }
        ]
    }
    '''
    send_data = handle_path(send_data)
    ego_path = []
    for i in range(len(send_data.x)):
        frameid = frame + i
        ego_path.append({
            "frameid": frameid,
            "info": [
                {
                    "id": "controlled",
                    "X": send_data.x[i] * 100,
                    "Y": send_data.y[i] * 100,
                    "yaw": send_data.yaw[i] * 180 / 3.1415926,
                    'Vx': send_data.Vx[i] * 100,
                    'Vy': send_data.Vy[i] * 100,
                    'ax': send_data.ax[i] * 100,
                    'ay': send_data.ay[i] * 100
                }]
            })
    result = {
        "V": ego_path
    }
    print(send_data.yaw[0:20:5])
    return result

def init_position():
    '''
    把被控车放到车道起点
    :return:
    '''
    path = FrenetPath()
    path.s = [20] * 5
    path.d = [0] * 5
    path.s_d = [5] * 5
    path.d_d = [0] * 5
    path.s_dd = [0] * 5
    path.d_dd = [0] * 5
    return path

def planner_server():
    # 创建socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定
    address = ('127.0.0.1', 6005)
    server_socket.bind(address)
    print("UDP服务器已启动")

    while True:
        # 接收数据
        data, client_address = server_socket.recvfrom(4096)  # 增加接收缓冲区大小
        print(f"收到来自{client_address}的数据")
        data = data.decode()

        # 场景开始时CAVE向服务端发送'S'字符
        if data == 'S':
            path = init_position()
            send_data = data_output_handle(1, path)
            server_socket.sendto(json.dumps(send_data).encode(), client_address)
            continue
        # 场景结束时CAVE向服务端发送'E'字符
        if data == 'E':
            break

        try:
            # 解析JSON数据
            recv_data = json.loads(data)
            print("解析JSON数据")
        except json.JSONDecodeError:
            print("接收到的数据不是有效的JSON格式")

        start_time = time.time()
        print("正在规划轨迹..........")
        data_input = data_input_handle(recv_data)
        if not data_input:
            print("no Controlled vehicle information")
            continue
        frame, recv_data = data_input[0], data_input[1]
        path, path_all = frenet_optimal_planning(csp, recv_data.s, recv_data.speed, recv_data.acc, recv_data.c_d, recv_data.c_d_d,
                                                             recv_data.c_d_dd, recv_data.ob, recv_data.ob_v)
        print("轨迹规划用了：",time.time() - start_time,"秒" )
        # 如果没有最优轨迹path，则从采样的轨迹中任意选择一个
        if path :
            send_data = path
        else:
            print("no best path")
            send_data = path_all[0]
##        send_data = init_position()

        # 格式转换
        send_data = data_output_handle(frame, send_data)
        end_time = time.time()
        print("一次轨迹规划从接收请求到发送结果一共用了：", end_time - start_time, "秒")

        # 发送数据
        client_address = ('127.0.0.1', 6004)
        send_data_str = json.dumps(send_data)
        server_socket.sendto(send_data_str.encode(), client_address)

        # plot 轨迹
        # plot_x = []
        # plot_y = []
        # for i in range(len(send_data['V'])):
        #     plot_x.append(send_data['V'][i]['info'][0]['X'])
        #     plot_y.append(send_data['V'][i]['info'][0]['Y'])
        # plt.plot(plot_x, plot_y)
        # plt.show()

if __name__ == '__main__':
    # 坐标转换需要
    global tx, ty, tyaw, csp
    tx, ty, tyaw, csp = Reference_line_build()
    planner_server()
    time.sleep(5)
