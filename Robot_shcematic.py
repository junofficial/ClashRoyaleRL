import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.axis('equal')
ax.set_xlim(-1, 2.5)
ax.set_ylim(-1, 2.5)
ax.grid(True)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Robot Movement')

Joint_1 = np.array([0, 0])
Joint_2 = np.array([1, 0])
Joint_3 = np.array([2, 0])

Robot_arm_1 = ax.plot([Joint_1[0], Joint_2[0]], [Joint_1[1], Joint_2[1]], linewidth=4)
Robot_arm_2 = ax.plot([Joint_2[0], Joint_3[0]], [Joint_2[1], Joint_3[1]], linewidth=4)

Robot_path = ax.plot([Joint_3[0]-0.01, Joint_3[0]], [Joint_3[1]-0.01, Joint_3[1]], linewidth=0.5)

plt.show()
plt.pause(0.1)
