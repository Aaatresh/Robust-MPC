"""
    Simulation of the linearized model by its forward propagation using a predefined input signal.
"""

""" Load necessary libraries """
# Obtain repository root for importing necessary libraries
from pathlib import Path
import sys
REPOROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOROOT))

import matplotlib.pyplot as plt
import system_config as servo_mech
from utils.utils import *


""" Load dynamics model """
# Load continuous time linear dynamics model
cont_lin_state_space = servo_mech.init_lin_dyn()
# Discretize continuous time linear model
disc_lin_state_space = cnt_to_dst(cont_lin_state_space, servo_mech.dt)


""" Lets simulate and visualize these dynamics """
x0 = np.zeros((4, 1))
t_array = np.arange(0, 5, servo_mech.dt)

# Build an input control signal to forward propogate dynamics model. Uncomment a control signal from below for use.
# u_array = 100 * np.exp(-np.power(t_array - 2.5, 2))  # Gaussian input
# u_array = 100 * np.exp(-2 * t_array)  # exponentially decreasing input
u_array = np.concatenate((np.ones((t_array.size // 2,)), np.zeros((t_array.size - (t_array.size // 2),))))  # step input

# Forward propagating the dynamics
xk_1 = x0
Y = []
for e, t_step in enumerate(t_array):
    xk = np.matmul(disc_lin_state_space["A"], xk_1) + (disc_lin_state_space["B"] * u_array[e])
    xk_1 = xk

    Y.append(np.matmul(disc_lin_state_space["C"], xk)[0])


""" Visualization """
# Visualizing the output and the input
fig, ax = plt.subplots(2, 1, figsize=(19.2, 10.8), sharex=True)
fig.suptitle("Plots of load angle (output) and voltage (input) versus time when simulating a servo-mechanical system's linear model")

ax[0].plot(t_array, Y)
ax[0].set_ylabel("Load angle (rad)")

ax[1].set_ylabel("Voltage input (V)")
ax[1].set_xlabel("Time (seconds)")
ax[1].plot(t_array, u_array)

plt.show()