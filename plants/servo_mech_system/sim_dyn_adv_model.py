"""
    Simulating non-linear dynamics by its forward propagation using a predefined input signal.
"""
""" Load necessary libraries """
import numpy as np
import matplotlib.pyplot as plt
import system_config as servo_mech

# Sampling time
T = 0.01

""" Lets simulate and visualize these dynamics """
x0 = np.zeros((4, 1))
t_array = np.arange(0, 5, T)

# Build an input control signal to forward propogate dynamics model. Uncomment a control signal from below for use.
u_array = 100 * np.exp(-np.power(t_array - 2.5, 2))  # Gaussian input
# u_array = 100 * np.exp(-2 * t_array)  # exponentially decreasing input
# u_array = np.concatenate((np.ones((t_array.size // 2,)), np.zeros((t_array.size - (t_array.size // 2),))))  # step input
# u_array = np.ones((t_array.size,)) # all ones input

# Forward propagating the dynamics
xk_1 = x0
Y = []
for e, t_step in enumerate(t_array):
    xk = xk_1 + (T * servo_mech.nonlin_dyn_step(xk_1, u_array[e], e + 1))
    xk_1 = xk

    Y.append(xk[0, 0])


""" Visualization """
# Visualizing the output and the input
fig, ax = plt.subplots(2, 1, figsize=(19.2, 10.8), sharex=True)
fig.suptitle("Plots of load angle (output) and voltage (input) versus time when simulating a servo-mechanical system's non-linear model")

ax[0].plot(t_array, Y)
ax[0].set_ylabel("Load angle (rad)")

ax[1].set_title("Voltage input")
ax[1].plot(t_array, u_array)
ax[1].set_ylabel("Voltage Input (V)")
ax[1].set_xlabel("Time (seconds)")

plt.show()
