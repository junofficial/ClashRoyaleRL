import numpy as np
from sys_params import SYS_PARAMS


global dt, initial_state, KpF, KdF, KpM, KdM
global old_pose, old_pose_d

params = SYS_PARAMS()
dt = params['Ts']

old_pose = []
old_pose_d = []
initial_state = [0, 0, 0.0998, 0, 0, 0, 0, 0, -1*np.pi/9.5, 0, 0, 0]


def UAV(x, u, params):
    dxdt = np.zeros(12)

    dxdt[0] = x[3]
    dxdt[1] = x[4]
    dxdt[2] = x[5]

    total_thrust = params['b'] / params['mass'] * sum(u)
    dxdt[3] = (np.cos(x[6]) * np.sin(x[7]) * np.cos(x[8]) +
               np.sin(x[6]) * np.sin(x[8])) * total_thrust
    dxdt[4] = (np.cos(x[6]) * np.sin(x[7]) * np.sin(x[8]) -
               np.sin(x[6]) * np.cos(x[8])) * total_thrust
    dxdt[5] = -params['grav'] + np.cos(x[6]) * np.cos(x[7]) * total_thrust

    dxdt[6] = x[9]
    dxdt[7] = x[10]
    dxdt[8] = x[11]

    dxdt[9] = params['Lxx'] * params['b'] / params['Ixx'] * \
        (u[3] - u[1]) + ((params['Iyy'] - params['Izz']) /
                         params['Ixx']) * x[9] * x[10]
    dxdt[10] = params['Lyy'] * params['b'] / params['Iyy'] * \
        (u[0] - u[2]) + ((params['Izz'] - params['Ixx']) /
                         params['Iyy']) * x[11] * x[9]
    dxdt[11] = params['d'] / params['Izz'] * (-u[0] + u[1] - u[2] + u[3]) + (
        (params['Ixx'] - params['Iyy']) / params['Izz']) * x[9] * x[10]

    return dxdt

def RK4(x_t: np.ndarray, v_t: np.ndarray, params) -> np.ndarray:
    x = x_t
    u = v_t
    k1 = dt * UAV(x, u, params)
    k2 = dt * UAV(x + 0.5 * k1, u, params)
    k3 = dt * UAV(x + 0.5 * k2, u, params)
    k4 = dt * UAV(x + k3, u, params)    
        
    x_updated = x + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x_updated    

def PID(pose, pose_d):
    global dt, initial_state
    global old_pose, old_pose_d
    params = SYS_PARAMS()

    KpF = np.array([4.55e0, 4.55e0, 2.55e0])
    KdF = np.array([8.75e0, 8.75e0, 2.75e0])
    KpM = np.array([3e2, 3e2, 3e2])
    KdM = np.array([3e1, 3e1, 3e1])

    if not len(old_pose):

        old_pose = np.concatenate(
            (initial_state[:3], initial_state[6:9]), axis=0)
    if not len(old_pose_d):
        old_pose_d = pose_d

    vx = (pose[0] - old_pose[0]) / dt
    vy = (pose[1] - old_pose[1]) / dt
    vz = (pose[2] - old_pose[2]) / dt
    vp = (pose[3] - old_pose[3]) / dt
    vq = (pose[4] - old_pose[4]) / dt
    vr = (pose[5] - old_pose[5]) / dt

    old_pose = pose

    dx_ref = (pose_d[0] - old_pose_d[0]) / dt
    dy_ref = (pose_d[1] - old_pose_d[1]) / dt
    dz_ref = (pose_d[2] - old_pose_d[2]) / dt
    dp_ref = (pose_d[3] - old_pose_d[3]) / dt
    dq_ref = (pose_d[4] - old_pose_d[4]) / dt
    dr_ref = (pose_d[5] - old_pose_d[5]) / dt

    old_pose_d = pose_d

    u1x = -KpF[0] * (pose[0] - pose_d[0]) - KdF[0] * (vx - dx_ref)
    u1y = -KpF[1] * (pose[1] - pose_d[1]) - KdF[1] * (vy - dy_ref)
    u1z = -KpF[2] * (pose[2] - pose_d[2]) - KdF[2] * \
        (vz - dz_ref) + params['grav']

    p_d = np.arctan2(np.sin(pose_d[5]) * np.cos(pose_d[5]) *
                     u1x - np.cos(pose_d[5]) * np.cos(pose_d[5]) * u1y, u1z)
    q_d = np.arcsin((np.cos(pose_d[5]) * np.cos(pose_d[5]) *
                    u1x + np.sin(pose_d[5]) * np.cos(pose_d[5]) * u1y) / u1z)
    r_d = pose_d[5]

    ang_d = np.array([p_d, q_d, r_d])
    rate_d = np.array([0, 0, 0])

    Fz = u1z / (np.cos(p_d) * np.cos(r_d))

    I = np.diag([params['Ixx'], params['Iyy'], params['Izz']])
    Mr = np.dot(
        I, KdM * (rate_d - np.array([vp, vq, vr])) + KpM * (ang_d - pose[3:6]))

    return Fz, Mr, ang_d
