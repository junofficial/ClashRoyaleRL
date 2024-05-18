import numpy as np

def SYS_PARAMS():
    m = 1.121
    g = 9.81
    b = 3e-6
    d = 1e-7
    c_m = 1e4
    Lxx = 0.2136
    Lyy = 0.1758
    uh = 0.538
    Kf = (m * g) / (4 * uh)
    Kt = 0.0045

    params = {
        'Ts': 0.01,
        'mass': m,
        'grav': g,
        'Lxx': Lxx,
        'Lyy': Lyy,
        'b': b * c_m,
        'd': d * c_m,
        'Ixx': 1.0e-2,
        'Iyy': 8.2e-3,
        'Izz': 1.48e-2,
        'Im': 1.2e-5,
        'Ax': 0.0025,
        'Ay': 0.0025,
        'Az': 0.0025,
        'Jr': 0.01,
        'maxangle': 40 * np.pi / 180,
        'maxF': 2.5 * m * g,
        'minF': 0.05 * m * g,
        'omegaMax': 330,
        'T2W': np.array([
            [b * c_m, b * c_m, b * c_m, b * c_m],
            [0, -Lxx * b * c_m, 0, Lxx * b * c_m],
            [Lyy * b * c_m, 0, -Lyy * b * c_m, 0],
            [-d * c_m, d * c_m, -d * c_m, d * c_m]
        ]),
        'W2I': np.array([
            [Kf, Kf, Kf, Kf],
            [-Kf * (Lxx / 2), -Kf * (Lxx / 2), Kf * (Lxx / 2), Kf * (Lxx / 2)],
            [Kf * (Lyy / 2), -Kf * (Lyy / 2), Kf * (Lyy / 2), -Kf * (Lyy / 2)],
            [Kt, -Kt, -Kt, Kt]
        ]),
        'N': 6,
        'bandwidth': 5
    }

    return params
