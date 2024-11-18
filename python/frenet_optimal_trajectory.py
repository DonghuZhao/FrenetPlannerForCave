"""
Frenet optimal trajectory generator
"""

import numpy as np
import copy
import math
import sys
import pathlib
import pickle
from quintic_polynomials_planner import QuinticPolynomial
import cubic_spline_planner
import csv
import time

# from mapplotfc import *
import shapely
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import unary_union

SIM_LOOP = 500

# Parameter
MAX_SPEED = 30.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 5.0  # maximum acceleration [m/ss]
MAX_DEACC = 5.0
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 2.0  # maximum road width [m]
D_ROAD_W = 0.5  # road width sampling length [m]                #0.5
DT = 0.1  # time tick [s]
MAX_T = 3.1  # max prediction time [m]                          #5.0
MIN_T = 3.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 3.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 5  # sampling number of target speed
ROBOT_RADIUS = 3.0  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 0.2 # 1.0
K_LAT = 1.0
K_LON = 1.0
Traj_plan = []
Traj_origin=[]


show_animation = False
SAVE_FLAG = False
time_data = time.localtime()

class Track_info:
    def __init__(self):
        self.object_type = None
        self.track_id = None
        self.timestep = []
        self.position = []
        self.velocity = []
        self.heading = []
        self.observed = []
        self.category = None
        self.track_type = None


def tracks_from_res(res):

    folder_name = res[0]
    left_veh_idx = res[1]
    straight_veh_idx = res[2]
    # leftturn_track =Track_info()
    # straight_track =Track_info()
    with open(f'./SinD_inter_data/{folder_name}/left_veh/{left_veh_idx}.data', 'rb') as filehandle:
        leftturn_track =pickle.load(filehandle)

    with open(f'./SinD_inter_data/{folder_name}/straight_veh/{straight_veh_idx}.data', 'rb') as filehandle:
        straight_track =pickle.load(filehandle)

    return leftturn_track, straight_track
def construct_line(x1, y1, x2, y2, x=None, y=None):
    # 计算两点之间的斜率
    if x1 == x2:
        k = None  # 竖直方向上的直线，斜率不存在
    else:
        k = (y2 - y1) / (x2 - x1)

    # 计算直线的截距
    b = y1 - k * x1 if k is not None else None

    if k is None:  # 竖直方向上的直线
        assert x is not None, "竖直方向上的直线必须给定 x 坐标"
        return x, y1
    else:
        assert x is None or y is None, "只能指定 x 或 y 中的一个"
        if x is not None:
            y = k * x + b
        else:
            x = (y - b) / k
        return x, y


def cal_traj_S_L_4_constrain(traj_points, left_polygon, midline_lane):
    # 计算S和L

    traj_l = []
    traj_s = []

    for idx, traj_point in enumerate(traj_points.geoms):
        # 轨迹点是否在left_polygon
        if left_polygon.contains(traj_point):
            flag_in = 'left'
        else:
            flag_in = 'right'

        # 轨迹点在中心线上的投影
        nearest_mid_point = midline_lane.interpolate(midline_lane.project(traj_point))

        #         nearest_mid_point = shapely.ops.nearest_points(traj_point,
        #                                                        midline_lane)[1]
        #         print(f'nearest_mid_point = {np.array(nearest_mid_point)}')
        trans_xs = (nearest_mid_point.x - traj_point.x) * 2
        trans_ys = (nearest_mid_point.y - traj_point.y) * 2
        trans_point = shapely.affinity.translate(traj_point, trans_xs, trans_ys)
        #         print(f'trans_point = {np.array(trans_point)}')
        vertical_line = LineString([traj_point, trans_point])
        mid_point = vertical_line.intersection(midline_lane)
        l_dis = traj_point.distance(mid_point)
        #         print(f'mid_point = {np.array(mid_point)}')
        #         print(f'traj_point = {np.array(traj_point)}')

        if flag_in == 'right':
            l_dis = l_dis
        else:
            l_dis = -l_dis

        polygon_split = shapely.ops.split(midline_lane, vertical_line)
        midline_lane_s = polygon_split.geoms[0]
        traj_s.append(midline_lane_s.length)
        traj_l.append(l_dis)

    return traj_s, traj_l


def cal_s_t_for_lefttrack(left_track):

    reference_line = refline_build()
    left_polygon = Polygon(np.vstack((reference_line, np.array([reference_line[0][0], reference_line[-1][1]]))))
    midline_lane = LineString(reference_line)
    traj_points = MultiPoint(np.array(left_track.position))
    traj_s, traj_l = cal_traj_S_L_4_constrain(traj_points, left_polygon, midline_lane)
    left_track.s = np.array(traj_s)
    left_track.l = - np.array(traj_l)

    s_speed = ( left_track.s[1:] - left_track.s[:-1] ) / DT
    l_speed = ( left_track.l[1:] - left_track.l[:-1] ) / DT
    s_acc = (s_speed[1:] - s_speed[:-1]) / DT
    l_acc = (l_speed[1:] - l_speed[:-1]) / DT
    s_jerk = (s_acc[1:] - s_acc[:-1]) / DT
    l_jerk = (l_acc[1:] - l_acc[:-1]) / DT

    left_track.s_speed = s_speed[2:]
    left_track.l_speed = l_speed[2:]
    left_track.s_acc = s_acc[1:]
    left_track.l_acc = l_acc[1:]
    left_track.s_jerk = s_jerk
    left_track.l_jerk = l_jerk

    left_track.s = left_track.s[3:]
    left_track.l = left_track.l[3:]
    left_track.timestep = np.array(left_track.timestep)[3:]
    left_track.position = np.array(left_track.position)[3:,:]
    left_track.velocity = np.array(left_track.velocity)[3:,:]
    left_track.heading = np.array(left_track.heading)[3:]

    return left_track

def refline_build():
    # 构造参考线
    P_ex = np.array([-26.44, 14.31])
    P_en = np.array([14.74, 43.67])
    P0 = np.array([-4.39, 14.54])
    P3 = np.array([14.85, 35.40])
    P1_x, P1_y = construct_line(P0[0], P0[1], P_ex[0], P_ex[1], x=10)
    P1 = np.array([P1_x, P1_y])
    P2_x, P2_y = construct_line(P3[0], P3[1], P_en[0], P_en[1], y=20)
    P2 = np.array([P2_x, P2_y])
    t = np.linspace(0, 1, 100)
    B = np.zeros((100, 2))
    for i in range(100):
        B[i] = (1 - t[i]) ** 3 * P0 + 3 * t[i] * (1 - t[i]) ** 2 * P1 + 3 * t[i] ** 2 * (1 - t[i]) * P2 + t[i] ** 3 * P3

    reference_line = np.vstack((P_ex, B))
    reference_line = np.vstack((reference_line, P_en))

    return reference_line

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # if s0 - 26.44 + 4.39 < -1:
    #     TARGET_SPEED = 5/3.6
    # else:
    #     TARGET_SPEED = 9/3.6

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning !!!!(Velocity keeping)!!!!
            for tv in np.arange(np.max([TARGET_SPEED - D_T_S * N_S_SAMPLE,0]),
                                np.min([TARGET_SPEED + D_T_S * N_S_SAMPLE,10]), D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob,v_ob):
    for i in range(len(fp.x)):

        d = [((ix - (ob[0] + v_ob[0]*j*DT)) ** 2 + (iy - ob[1] ) ** 2)
             for (ix, iy, j) in zip(fp.x, fp.y, range(0, len(fp.x)))]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            # print('Collision happened!')
            return False

    return True

# def check_exitlane():


def check_paths(fplist, ob, v_ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        # if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
        #     continue
        if any([ MAX_DEACC < a < MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue

        # elif any([abs(a) > MAX_ACCEL for a in
        #           fplist[i].s_dd]):  # Max accel check
        #     continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob,v_ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob,v_ob):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist_all = calc_global_paths(fplist, csp)
    fplist_check = check_paths(fplist_all, ob, v_ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist_check:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path,fplist_all


def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def main(left_track, straight_track,left_start_idx, idx,ax):

    t0 = time.time()

    # 构造中心线
    P_ex = np.array([-26.44, 14.31])
    P_en = np.array([14.74, 43.67])
    P0 = np.array([-4.39, 14.54])
    P3 = np.array([14.85, 35.40])
    P1_x, P1_y = construct_line(P0[0], P0[1], P_ex[0], P_ex[1], x=10)
    P1 = np.array([P1_x, P1_y])
    P2_x, P2_y = construct_line(P3[0], P3[1], P_en[0], P_en[1], y=20)
    P2 = np.array([P2_x, P2_y])
    t = np.linspace(0, 1, 100)
    B = np.zeros((100, 2))
    for i in range(100):
        B[i] = (1 - t[i]) ** 3 * P0 + 3 * t[i] * (1 - t[i]) ** 2 * P1 + 3 * t[i] ** 2 * (1 - t[i]) * P2 + t[i] ** 3 * P3

    reference_line = np.vstack((P_ex, B))
    reference_line = np.vstack((reference_line, P_en))

    wx = reference_line[:, 0]
    wy = reference_line[:, 1]
    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    t1=time.time()
    print(t1-t0)
    print("typeofcsp.s:",type(csp.s),csp.s)
    print("typeofcsp.sx:",type(csp.sx),csp.sx)
    print("typeofcsp.sy:",type(csp.sy),csp.sy)



    # 计算初始S和L
    left_polygon = Polygon(np.vstack((reference_line, np.array([P_ex[0], P_en[1]]))))
    midline_lane = LineString(reference_line)


    ### 开始读数据，存在left_track和straight_track结构体中了，结构体的定义见class Track_info
    ### 可以实时地print看一下
    ### 在循环外面读的是初始状态
    ### 我们需要读的数据：
    ### 左转车初始状态，也就是场景的初始状态设置
    ### 直行车每一时刻的位置(x,y)和速度(vx,vy) --->需要写进下面for i in range(SIM_LOOP)循环中
    ### 有以上信息传入之后，整个程序就会运行起来
    ### 左转车规划的结果在path,path_all = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob, v_ob)这句函数的输出
    ### 其中path为class FrenetPath结构体，path为最优轨迹，path.x[1]和path.y[1]为下一时刻的规划位置，需要传给仿真环境

    traj_points = MultiPoint(left_track.position[left_start_idx:left_start_idx + 2])
    # 轨迹规划中，左为正右为负！之前的分析过程左负右正！！需要加负号
    traj_s, traj_l = cal_traj_S_L_4_constrain(traj_points, left_polygon, midline_lane)

    # 获取左转车和直行车的初始状态
    start_timestep = left_track.timestep[left_start_idx]
    straight_start_idx = np.where(np.array(straight_track.timestep) == start_timestep)[0]

    if len(straight_start_idx) == 0:
        straight_start_idx = 0
    straight_indices = np.arange(straight_start_idx, len(straight_track.position), int(DT * 10))
    straight_position_np = np.array(straight_track.position)[straight_indices]
    straight_velocity_np = np.array(straight_track.velocity)[straight_indices]

    left_velocity = left_track.velocity[left_start_idx]
    left_acc = (left_track.velocity[left_start_idx + 1] - left_track.velocity[left_start_idx]) / 0.1

    # initial state---纵向
    c_speed = left_velocity[0]  # 10.0 / 3.6  # current speed [m/s]
    c_accel = left_acc[0]  # current acceleration [m/ss]
    s0 = traj_s[0]
    # print(s0)
    # initial state---横向
    c_d = -traj_l[0]  # current lateral position [m]
    c_d_d = left_velocity[1]  # current lateral speed [m/s]
    c_d_dd = left_acc[1]  # current lateral acceleration [m/s]
    # area = 20.0  # animation area length [m]

    for i in range(SIM_LOOP):
        print(i)
        # ob = ob + deta_ob
        if i < len(straight_position_np):
            ob = np.array([straight_position_np[i,:]])
            v_ob = np.array([straight_velocity_np[i, :]])
        else:
            pass
        # ob = ob + v_ob * DT
        path,path_all = frenet_optimal_planning(
            csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob, v_ob)

        # print("path.s",path.s)
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]
        if i == 0:
            left_trajplan_np = np.array([path.x[1],path.y[1]])
        else:
            left_trajplan_np = np.vstack((left_trajplan_np, np.array([path.x[1], path.y[1]])))

        # print(s0)
        if (np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0) or \
                (path.y[2]>=34):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            map_path = r'D:\驾驶模拟实验2\Trajectory_planning\Trajectory_planning\SinD_inter_data\mapfile-Tianjin_revised0224.osm'
            draw_map_without_lanelet_original(map_path, ax, 0, 0)
            # plt.plot(tx, ty)
            for fp_plt in path_all:
                plt.plot(fp_plt.x,fp_plt.y,color='gray',alpha=0.2)

            plt.scatter(np.array(left_track.position)[0:2*i, 0], np.array(left_track.position)[0:2*i, 1],
                        s=5,c='C0',marker='s',alpha=0.5,zorder=11)
            plt_rec(ax, left_track.position[min(2*i,len(left_track.position)-1)][0],left_track.position[min(2*i,len(left_track.position)-1)][1],
                    color_veh='C0', heading=left_track.heading[min(2*i,len(left_track.position)-1)], obj='left')
            plt_rec(ax, ob[:,0], ob[:,1], color_veh='C1', obj='straight')
            # plt.plot(ob[0], ob[1], "xk",zorder=12)
            plt_rec(ax, path.x[0], path.y[0], color_veh='red', heading=path.yaw[0], obj='left')
            plt.plot(path.x[1:], path.y[1:], c="red", zorder=12)
            if i >=1 :
                Traj_plan.append([path.x[0],path.y[0],path.s[0],path.d[0],path.yaw[0]])
                plt.scatter(np.array(Traj_plan)[:,0],np.array(Traj_plan)[:,1],s=5,c='red',alpha=0.8,zorder=12)
                Traj_origin.append([left_track.position[min(2*i,len(left_track.position)-1)][0],left_track.position[min(2*i,len(left_track.position)-1)][1],
                                    left_track.s[min(2*i,len(left_track.position)-1)],left_track.l[min(2*i,len(left_track.position)-1)],
                                    left_track.heading[min(2*i,len(left_track.position)-1)]])
            # plt.plot(ob[:, 0], ob[:, 1], "xk")
            # plt.plot(path.x[1:], path.y[1:], "-or")
            # plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim([-30, 60])
            plt.ylim([-12, 45])
            # plt.xlim(path.x[1] - area, path.x[1] + area)
            # plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title(f"T[s]:{2*i/10}  v[m/s]:{str(c_speed)[0:4]}  Planning Trajectory Num:{len(path_all)}")
            # plt.grid(True)
            plt.pause(0.0001)
            if SAVE_FLAG :
                plt.savefig(directory +'/'+str(i)+'.png')

    t1 = time.time()
    print(f"Scenario {idx} Finish, time needed {round(t1-t0)} s")

    # with open(f'./pathplanning_result/Traj_plan_nonopt_PREEMPT.data', 'wb') as filehandle:
    #     pickle.dump(Traj_plan, filehandle)

    if show_animation:  # pragma: no cover
        # plt.grid(True)
        plt.pause(0.0001)
        plt.show()

    return left_trajplan_np

if __name__ == '__main__':

    filePath = 'D:/文件/驾驶模拟实验2/Trajectory_planning/Trajectory_planning/SinD_inter_data/res_SinD_revised_manu.csv'
    col_types = [str, int, int, int, int, int, int, str]
    res_SinD_manu = []
    with open(filePath, "r") as file:
        data = csv.reader(file)
        for row in data:
            row = list(tuple(convert(value) for convert, value in zip(col_types, row)))
            res_SinD_manu.append(row)

    # map_path = r'E:\OneDrive - tongji.edu.cn\博士毕业论文\GametheoryBehavioranalysis\PyProject\SinD_main\Data\mapfile-Tianjin_revised0224.osm'
    # fig, ax = plt.subplots()
    # draw_map_without_lanelet_original(map_path,ax, 0, 0)
    fig, ax = plt.subplots()
    for idx, res_temp in enumerate(res_SinD_manu):
        res = res_temp
        if idx == 79: #7
            print(f'idx={idx},res={res}')
            start_timestep = res[6]
            PET = res[3]
            left_track, straight_track = tracks_from_res(res)
            left_track = cal_s_t_for_lefttrack(left_track)
            directory = f'./pathplanning_result/fig_scenario/nonoptimal_{idx}_{time_data[1]}_{time_data[2]}_{time_data[3]}_{time_data[4]}'
            if SAVE_FLAG:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            main(left_track, straight_track,start_timestep,idx,ax)


"""

if __name__ == '__main__':

    filePath = './SinD_inter_data/res_SinD_revised_manu.csv'
    col_types = [str, int, int, int, int, int, int, str]
    res_SinD_manu = []
    with open(filePath, "r") as file:
        data = csv.reader(file)
        for row in data:
            row = list(tuple(convert(value) for convert, value in zip(col_types, row)))
            res_SinD_manu.append(row)

    left_trajplan_all=[]

    preempt_num = 0
    yield_num = 0
    error_num = 0

    map_path = r'E:\OneDrive - tongji.edu.cn\博士毕业论文\GametheoryBehavioranalysis\PyProject\SinD_main\Data\mapfile-Tianjin_revised0224.osm'
    fig, ax = plt.subplots()
    draw_map_without_lanelet_original(map_path,ax, 0, 0)


    for idx, res_temp in enumerate(res_SinD_manu):
        res = res_temp
        left_trajplan_np = []
        plt_flag = True
        if res[7] == 'w':
            print(f'idx={idx},res={res}')
            start_timestep = res[6]
            PET = res[3]
            # if preempt_num >=15 and yield_num >= 15:
            #     break
            # elif PET > 0 and preempt_num < 15 :
            #     preempt_num = preempt_num + 1
            #     print(f'第{preempt_num}个先行轨迹...')
            # elif PET < 0 and yield_num < 15 :
            #     yield_num = yield_num + 1
            #     print(f'第{yield_num}个让行轨迹...')
            # else:
            #     continue

            if PET > 0 :
                preempt_num = preempt_num + 1
                print(f'第{preempt_num}个先行轨迹...')
            elif PET < 0 :
                yield_num = yield_num + 1
            left_track, straight_track = tracks_from_res(res)
            try:
                left_trajplan_np = main(left_track, straight_track,start_timestep,idx)
            except:
                error_num = error_num  + 1
                print(f'{error_num}个错误！')
                plt_flag = False

            if plt_flag:
                # plt.cla()
                # for stopping simulation with the esc key.
                # plt.gcf().canvas.mpl_connect(
                #     'key_release_event',
                #     lambda event: [exit(0) if event.key == 'escape' else None])
                if PET > 0 :
                    plt.plot(left_trajplan_np[:,0], left_trajplan_np[:,1],c='C0',alpha=0.2)
                else:
                    plt.plot(left_trajplan_np[:, 0], left_trajplan_np[:, 1], c='C1', alpha=0.2)
                # plt.grid(True)
                plt.pause(0.0001)
        left_trajplan_all.append(left_trajplan_np)

    with open(f'./pathplanning_result/left_trajplan_all.data', 'wb') as filehandle:
        pickle.dump(left_trajplan_all, filehandle)

    # plt.grid(True)
    plt.pause(0.0001)
    plt.show()

"""


