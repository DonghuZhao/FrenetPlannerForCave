import numpy as np
from config import *
import math

def projection(x, y, tx, ty, tyaw, csp):
    dist_x = [x - value for value in tx]
    dist_y = [y - value for value in ty]
    dist = [math.sqrt(x ** 2 + y ** 2) for x, y in zip(dist_x, dist_y)]
    min_dist = min(dist)
    return min_dist, dist.index(min_dist)

def Cartesian_to_Frenet(x, y, vx, vy, ax, ay, yaw):
    global tx, ty, tyaw, csp
    # 计算投影点
    min_dist, min_index = projection(x, y, tx, ty, tyaw, csp)
    rx = tx[min_index]
    ry = ty[min_index]
    rtheta = tyaw[min_index]
    dx = x - rx
    dy = y - ry
    if dy * math.cos(rtheta) - dx * math.sin(rtheta) > 0:
        l_sign = 1.0
    else:
        l_sign = -1.0
    l = l_sign * min_dist
    print(min_index, len(csp.s))
    s = csp.s[min_index]
    vtheta = math.atan2(vy, vx)
    speed = math.sqrt(vx ** 2 + vy ** 2)
    s_d = speed * math.cos(vtheta - rtheta)
    l_d = speed * math.sin(vtheta - rtheta)
    acce = math.sqrt(ax ** 2 + ay ** 2)
    s_dd = acce * math.cos(vtheta - rtheta)
    l_dd = acce * math.sin(vtheta - rtheta)
    return s, s_d, s_dd, l, l_d, l_dd


def Frenet_to_Cartesian(csp,s,s_d,s_dd,d,d_d,d_dd):
    fx,fy = csp.calc_position(s)
    fyaw = csp.calc_yaw(s)
    editx = fx + d * math.cos(fyaw + math.pi/2.0)
    edity = fy + d * math.sin(fyaw + math.pi/2.0)
    editvx = s_d * math.cos(fyaw) + d_d * math.cos(fyaw + math.pi/2.0)
    editvy = s_d * math.sin(fyaw) + d_d * math.sin(fyaw + math.pi/2.0)
    editax = s_dd * math.cos(fyaw)+ d_dd * math.cos(fyaw + math.pi/2.0)
    editay = s_dd * math.sin(fyaw) + d_dd * math.sin(fyaw + math.pi/2.0)
    edityaw = math.atan2(editvy,editvx)
    return [editx,edity,edityaw,editvx,editvy,editax,editay]


if __name__ == "__main__":
    global tx, ty, tyaw, csp
    tx, ty, tyaw, csp = Reference_line_build()