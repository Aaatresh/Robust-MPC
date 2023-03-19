"""
    Run the robust MPC in different scenarios. For more information about the scenarios, refer README.md.
"""

# Importing necessary libraries

## To attach root of this repository to PYTHONPATH
import sys
import os
from pathlib import Path
REPOROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPOROOT))

## For data visualization and plotting
import numpy as np
import matplotlib.pyplot as plt

## Import controller, controller configuration, and plant configuration
from libs.controllers.robust_mpc_def import robust_mpc
# from libs.controllers.controller_config import config_params
from plants.servo_mech_system import system_config as servo_system

## Import utils
from utils.utils import cnt_to_dst, load_yaml

## Import setpoint generation function
from plants.servo_mech_system.setpoint_generator import const_setpoint_gen

## Import argument parser and setup
import argparse
parser = argparse.ArgumentParser("Run robust MPC in different scenarios. For information about scenarios, refer to README.md.")
parser.add_argument("-s", type=int, default=None, required=True, help="1 or 2, corresponding to scenario 1 or 2.")
parser.add_argument("--savepath",type=str, default=None, help="Directory in which plots are to be saved.")
parser.add_argument("--controller_config_filepath", type=str, default=None, required=True, help="Path to YAML controller configuration file.")
args = parser.parse_args()

# Check value of '-s' argument
if(not (args.s == 1 or args.s == 2)):
    raise ValueError("\'-s\' argument must be either 1 or 2.")

CONTROLLER_NAME = "robust"
controller_config_params = load_yaml(args.controller_config_filepath)

# Convert the model to discrete-time
cont_lin_state_space = servo_system.init_lin_dyn()
disc_lin_state_space = cnt_to_dst(cont_lin_state_space, servo_system.model_params["dt"])

# Define a constant setpoint value. This can be changed based on the type of setpoint tracking desired. This signal
# can be made more complex as well.
r = np.pi / 2


# Weight matrices Rk and Qk
Rk = controller_config_params[CONTROLLER_NAME][args.s]["Rk"]
Qk = controller_config_params[CONTROLLER_NAME][args.s]["Qk"]

# Initial state covariance and mean
init_Pt = controller_config_params[CONTROLLER_NAME][args.s]["init_Pt_std"] * np.eye(disc_lin_state_space["A"].shape[0])
init_xtt_1 = controller_config_params[CONTROLLER_NAME][args.s]["init_xtt_std"] * \
             np.random.randn(disc_lin_state_space["A"].shape[0], 1)


# Initialize yt
y0 = np.array([[0]])
next_yt = y0

# Simulation time parameters: TODO: Parameterize upper bound on time steps
tspan = [0, 20]

all_Ys = []
all_Us = []
all_covs = []

t_array = np.arange(tspan[0], tspan[1], servo_system.model_params["dt"])

# controller
controller = robust_mpc(disc_lin_state_space, controller_config_params, Rk, Qk, init_Pt, init_xtt_1)

# Simulation loop
for e, t in enumerate(t_array):

    # collect new data
    yt = next_yt

    # Define setpoint for a certain prediction horizon
    rt = const_setpoint_gen(r, controller_config_params["Hp"])

    # Calculate optimal control input
    xtt, utt = controller.step(yt, rt)

    # Apply utt to the system
    if(args.s == 1):
        vtt = np.random.randn(disc_lin_state_space["A"].shape[0], 1)
        next_xtt = np.matmul(disc_lin_state_space["A"], xtt) + (disc_lin_state_space["B"] * utt) + np.matmul(controller.G1, vtt)
        next_yt = np.matmul(disc_lin_state_space["C"], next_xtt) + np.matmul(controller.D1, vtt)
    elif(args.s == 2):
        next_xtt = xtt + (servo_system.model_params["dt"] * servo_system.nonlin_dyn_step(xtt, utt, e + 1))
        next_yt = np.matmul(disc_lin_state_space["C"], next_xtt)

    # Store variables for plotting
    all_Us.append(utt[0, 0])
    all_Ys.append(yt[0, 0])
    all_covs.append(controller.Vt[0, 0])


plt.figure()
plt.plot(t_array, all_Ys, label='Output')
plt.axhline(y=r, color='k', linestyle='--', label='Set-point= (pi / 2)')
plt.fill_between(t_array, y1=[y + np.sqrt(c) for y, c in zip(all_Ys, all_covs)],
                 y2=[y - np.sqrt(c) for y, c in zip(all_Ys, all_covs)], alpha=0.5)
plt.title("Plot of output versus time")
plt.ylabel("Angle (rad)")
plt.xlabel("Time (sec)")
plt.legend()

if(args.savepath is not None):
    plt.savefig(os.path.join(args.savepath, f"y_vs_t_{CONTROLLER_NAME}_MPC_Scenario_{args.s}.png"))

plt.figure()
plt.plot(all_Us)
plt.title("Plot of control input versus time")
plt.ylabel("Input voltage (V)")
plt.xlabel("Time (sec)")

if(args.savepath is not None):
    plt.savefig(os.path.join(args.savepath, f"u_vs_t_{CONTROLLER_NAME}_MPC_Scenario_{args.s}.png"))

plt.show()