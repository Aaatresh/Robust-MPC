"""
    Run the robust MPC in different scenarios. For more information about the scenarios, refer README.md.
"""

## Attach root of this repository to PYTHONPATH
import sys
from pathlib import Path
REPOROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPOROOT))

import os
import numpy as np
import matplotlib.pyplot as plt

from libs.mpc_controllers.validate import validate_control_params

from libs.mpc_controllers.robust_mpc_def import robust_mpc
from plants.servo_mech_system.system_config import servo_mech_plant
from utils.utils import visualize_controller, load_yaml                             ## Import utility functions
from plants.servo_mech_system.setpoint_generator import const_setpoint_gen          # Import setpoint generation function


## Import argument parser and setup
import argparse
parser = argparse.ArgumentParser("Run robust MPC in different scenarios. For information about scenarios, refer to README.md.")
parser.add_argument("-s", type=int, default=None, required=True, help="1 or 2, corresponding to scenario 1 or 2.")
parser.add_argument("--savepath",type=str, default=None, help="Directory in which plots are to be saved.")
parser.add_argument("--controller_config_filepath", type=str, default=None, required=True, help="Path to YAML controller configuration file.")
parser.add_argument("--plant_config_filepath", type=str, default=None, required=True, help="Path to YAML plant configuration file.")
args = parser.parse_args()

# Check value of '-s' argument
if(not (args.s == 1 or args.s == 2)):
    raise ValueError("\'-s\' argument must be either 1 or 2.")
if(args.savepath is not None and not os.path.isdir(args.savepath)):
    raise ValueError("Directory provided to store plots does not exist.")
if(not os.path.exists(args.controller_config_filepath)):
    raise ValueError("Controller configuration YAML file does not exist.")
elif(args.controller_config_filepath.split(".")[-1].lower() != "yaml"):
    raise ValueError("Controller configuration file must be of YAML format. Please make sure that the file name has the \".yaml\" extension.")

CONTROLLER_NAME = "robust"
controller_config_params = validate_control_params(load_yaml(args.controller_config_filepath))


# Initialize servo-mechanical system
servo_system = servo_mech_plant(args.plant_config_filepath)


# Define a constant setpoint value. This can be changed based on the type of setpoint tracking desired. This signal
# can be made more complex as well.
r = np.pi / 2


# Weight matrices Rk and Qk
Rk = controller_config_params[CONTROLLER_NAME][args.s]["Rk"]
Qk = controller_config_params[CONTROLLER_NAME][args.s]["Qk"]

# Initial state covariance and mean
init_Pt = controller_config_params[CONTROLLER_NAME][args.s]["init_Pt_std"] * np.eye(servo_system.disc_lin_state_space["A"].shape[0])
init_xtt_1 = controller_config_params[CONTROLLER_NAME][args.s]["init_xtt_std"] * \
             np.random.randn(servo_system.disc_lin_state_space["A"].shape[0], 1)


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
controller = robust_mpc(servo_system.disc_lin_state_space, controller_config_params, Rk, Qk, init_Pt, init_xtt_1,
                        [-servo_system.model_params["Vmax"], servo_system.model_params["Vmax"]])

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
        next_xtt, next_yt = servo_system.fwd_prop_lin_model(xtt, utt, controller.G1, controller.D1)
    elif(args.s == 2):
        next_xtt, next_yt = servo_system.fwd_prop_nonlin_model(xtt, utt, e)

    # Store variables for plotting
    all_Us.append(utt[0, 0])
    all_Ys.append(yt[0, 0])
    all_covs.append(controller.Vt[0, 0])


# Visualize result of simulation
collected_data = [t_array, all_Ys, all_Us, all_covs]
visualize_controller(collected_data, args, CONTROLLER_NAME, r)

plt.show()