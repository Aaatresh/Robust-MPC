"""
    Run the vanilla MPC in different scenarios. For more information about the scenarios, refer README.md.
"""

# Importing necessary libraries

## To attach root of this repository to PYTHONPATH
import sys
from pathlib import Path
REPOROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPOROOT))

## For data visualization and plotting
import matplotlib.pyplot as plt

## Import controller, controller configuration, and plant configuration
from libs.controllers.controller_config import *
from libs.servo_mech_system import system_config as servo_system
from libs.controllers.vanilla_mpc_def import vanilla_mpc

## Import utils
from utils.utils import *

## Import setpoint generation function
from libs.servo_mech_system.setpoint_generator import const_setpoint_gen

## Import argument parser and setup
import argparse
parser = argparse.ArgumentParser("Run vanilla MPC in different scenarios. For information about scenarios, refer to README.md.")
parser.add_argument("-s", type=int, default=None, required=True, help="1 or 2, corresponding to scenario 1 or 2.")
args = parser.parse_args()

# Check value of '-s' argument
if(not (args.s == 1 or args.s == 2)):
    raise ValueError("\'-s\' argument must be either 1 or 2.")


# Convert the model to discrete-time
A, B, C = cnt_to_dst(servo_system.Ac, servo_system.Bc, servo_system.C, servo_system.dt)

# Define a constant setpoint value. This can be changed based on the type of setpoint tracking desired. This signal
# can be made more complex as well.
r = np.pi / 2


# Initialize yt
y0 = np.array([[0]])
next_yt = y0

# Simulation time parameters
tspan = [0, 20]

if(args.s == 1):
    # Weight matrices Rk and Qk
    Rk = 0.1
    Qk = 1e4

    # Initial state covariance and mean
    init_Pt = 1e-4 * np.eye(A.shape[0])
    init_xtt_1 = 3e-2 * np.random.randn(A.shape[0], 1)

elif(args.s == 2):
    # Weight matrices Rk and Qk
    Rk = 0.01
    Qk = 5e3

    # Initial state covariance and mean
    init_Pt = np.eye(A.shape[0])
    init_xtt_1 = 1e-2 * np.random.randn(A.shape[0], 1)


# Create empty lists to store relevant variables at teach time step
all_Ys = []
all_Us = []
all_covs = []

t_array = np.arange(tspan[0], tspan[1], servo_system.dt)

# controller
controller = vanilla_mpc(A, B, C, Rk, Qk, Hu, Hp, act_model_std, sen_model_std, init_Pt, init_xtt_1)

# Simulation loop
for e, t in enumerate(t_array):

    # collect new data
    yt = next_yt

    # Define setpoint for a certain prediction horizon
    rt = const_setpoint_gen(r, Hp)

    # Calculate optimal control input
    xtt, utt = controller.step(yt, rt)


    # Apply utt to the system
    if(args.s == 1):
        vtt = np.random.randn(A.shape[0], 1)
        next_xtt = np.matmul(A, xtt) + (B * utt) + np.matmul(controller.G1, vtt)
        next_yt = np.matmul(C, next_xtt) + np.matmul(controller.D1, vtt)
    elif(args.s == 2):
        next_xtt = xtt + (servo_system.dt * servo_system.non_lin_dyn(xtt, utt, e + 1))
        next_yt = np.matmul(C, next_xtt)


    # Store variables for plotting
    all_Ys.append(yt[0, 0])
    all_covs.append(controller.Pt[0, 0])
    all_Us.append(utt[0, 0])


plt.figure()
plt.plot(t_array, all_Ys, label='Output')
plt.axhline(y=r, color='k', linestyle='--', label='Set-point= (pi / 2)')
plt.fill_between(t_array, y1=[y + np.sqrt(c) for y, c in zip(all_Ys, all_covs)],
                 y2=[y - np.sqrt(c) for y, c in zip(all_Ys, all_covs)], alpha=0.5)
plt.title("Plot of output versus time")
plt.ylabel("Angle (rad)")
plt.xlabel("Time (sec)")
plt.legend()

plt.figure()
plt.plot(all_Us)
plt.title("Plot of control input versus time")
plt.ylabel("Input voltage (V)")
plt.xlabel("Time (sec)")

plt.show()