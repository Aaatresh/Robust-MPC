"""
Simulation of the linearized model
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
L = 0  # armature coil inductance
# J_m = 0.5  # motor inertia
J_m = 0.5  # motor inertia
beta_m = 0.1   # motor viscous friction coefficient
R = 20   # resistance of armature
Kt = 10  # motor constant
rho = 20  # gear ratio
k_theta = 1280.2  # torsional rigidity
J_l = 25  # nominal load inertia
# J_l = 10  # nominal load inertia
beta_l = 25  # load viscous friction coefficient

del_t = 0.1

# Define A
Ac = np.array([
    [0, 1, 0, 0],
    [-(k_theta) / J_l, -beta_l / J_l, (k_theta) / (rho * J_l), 0],
    [0, 0, 0, 1],
    [(k_theta) / (rho * J_m), 0, (-k_theta) / ((rho**2) * J_m),  - ((Kt**2) / (J_m * R)) - (beta_m / J_m)]
])

# Define B
Bc = np.array([
    [0],
    [0],
    [0],
    [Kt / (J_m * R)]
])

import scipy.linalg

# Convert the model to discrete-time
T = 0.01
A = scipy.linalg.expm(Ac*T)
B = np.matmul(np.linalg.pinv(Ac), np.matmul((scipy.linalg.expm(Ac*T)-np.eye(4, 4)), Bc))

# Define C
C = np.array([
    [1, 0, 0, 0]
])

""" Lets simulate and visualize these dynamics """

x0 = np.zeros((4, 1))
t_array = np.arange(0, 5, T)
# u_array = 100 * np.exp(-np.power(t_array - 2.5, 2))  # Gaussian input
# u_array = 100 * np.exp(-2 * t_array)  # exponentially decreasing input
u_array = np.concatenate((np.ones((t_array.size // 2,)), np.zeros((t_array.size - (t_array.size // 2),))))  # step input

# Visualizing the input
# plt.figure()
# plt.title("Voltage input")
# plt.plot(t_array, u_array)
# plt.show()
# exit()

# Forward propagating the dynamics
xk_1 = x0
Y = []
for e, t_step in enumerate(t_array):
    xk = np.matmul(A, xk_1) + (B * u_array[e])
    xk_1 = xk

    Y.append(xk[0, 0])

# Visualizing the output and the input
fig, ax = plt.subplots(2, 1)
ax[0].set_title("Load angle")
ax[0].plot(Y)

ax[1].set_title("Voltage input")
ax[1].plot(u_array)

plt.show()