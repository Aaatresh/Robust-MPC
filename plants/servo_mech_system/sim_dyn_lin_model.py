"""
    Simulation of the linearized or non-linearized model through its forward propagation using a predefined input signal.
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
import argparse

parser = argparse.ArgumentParser("Simulation of linear or non-linear dynamics of a servo-mechanical system.")
parser.add_argument("--nonlin", type=int, required=True, help="Simulate non-linear dynamics? If yes, pass 1, else pass 0.")
args = parser.parse_args()

if(not (args.nonlin == 0 or args.nonlin == 1)):
    raise ValueError("Argument for --nonlin must be either an integer with value 1 or 0.")


""" Load dynamics model """
# Load continuous time linear dynamics model
cont_lin_state_space = servo_mech.init_lin_dyn()
# Discretize continuous time linear model
disc_lin_state_space = cnt_to_dst(cont_lin_state_space, servo_mech.dt)

# Sampling time
T = 0.01


""" Lets simulate and visualize these dynamics """
x0 = np.zeros((4, 1))
if(args.nonlin):
    t_array = np.arange(0, 5, T)
else:
    t_array = np.arange(0, 5, servo_mech.dt)

# Build an input control signal to forward propogate dynamics model. Uncomment a control signal from below for use.
u_array = 100 * np.exp(-np.power(t_array - 2.5, 2))  # Gaussian input
# u_array = 100 * np.exp(-2 * t_array)  # exponentially decreasing input
# u_array = np.concatenate((np.ones((t_array.size // 2,)), np.zeros((t_array.size - (t_array.size // 2),))))  # step input

# Forward propagating the dynamics
xk_1 = x0
Y = []
for e, t_step in enumerate(t_array):

    if(args.nonlin):
        xk = xk_1 + (T * servo_mech.nonlin_dyn_step(xk_1, u_array[e], e + 1))
    else:
        xk = np.matmul(disc_lin_state_space["A"], xk_1) + (disc_lin_state_space["B"] * u_array[e])

    xk_1 = xk

    Y.append(np.matmul(disc_lin_state_space["C"], xk)[0])


""" Visualization """
linearity = "non-linear" if args.nonlin else "linear"

# Visualizing the output and the input
fig, ax = plt.subplots(2, 1, figsize=(19.2, 10.8), sharex=True)
fig.suptitle(f"Plots of load angle (output) and voltage (input) versus time when simulating a servo-mechanical system's {linearity} model")

ax[0].plot(t_array, Y)
ax[0].set_ylabel("Load angle (rad)")

ax[1].set_ylabel("Voltage input (V)")
ax[1].set_xlabel("Time (seconds)")
ax[1].plot(t_array, u_array)

plt.show()