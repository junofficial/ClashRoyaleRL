import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 임의의 데이터 생성
n_samples = 100

# 초기 데이터
sampled_traj_x_offset = np.linspace(0, np.pi, n_samples)
sampled_traj_y_offset = np.linspace(0, np.pi / 2, n_samples)

fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', label='Sample Data')

def init():
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Real-time Data Visualization')
    ax.legend()
    ax.grid(True)
    return line,

def update(frame):
    global sampled_traj_x_offset, sampled_traj_y_offset

    # 데이터 업데이트 (예시로 간단하게 변화를 줌)
    sampled_traj_x_offset += 0.01
    sampled_traj_y_offset += 0.01

    # 새로운 x_sample 및 y_sample 계산
    x_sample = [1 * np.cos(q1) + 1 * np.cos(q1 + q2) for (q1, q2) in zip(sampled_traj_x_offset, sampled_traj_y_offset)]
    y_sample = [1 * np.sin(q1) + 1 * np.sin(q1 + q2) for (q1, q2) in zip(sampled_traj_x_offset, sampled_traj_y_offset)]

    # 그래프 데이터 업데이트
    line.set_data(x_sample, y_sample)
    return line,

# 애니메이션 생성
ani = FuncAnimation(fig, update, frames=range(200), init_func=init, blit=True, interval=100)

plt.show()

