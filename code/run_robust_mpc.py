"""
    Performance of the robust MPC when the approximated model and the actual model coincide
"""

# Importing necessary libraries
import matplotlib.pyplot as plt
from controller_config import *
import system_config as servo_system
from utils import *
from robust_mpc_def import robust_mpc

"""
Choose from scenario 1 and 2:
"""
situation = "scenario 1"

# Check value of 'situation'
if(not (situation == "scenario 1" or situation == "scenario 2")):
    raise ValueError("\'situation\' variable must be either \"scenario 1\" or \"scenario 2\".")


# Convert the model to discrete-time
A, B, C = cnt_to_dst(servo_system.Ac, servo_system.Bc, servo_system.C, servo_system.dt)

# Set point
r = np.pi / 2
rt = r * np.ones((Hp, 1))

if(situation == "scenario 1"):
    # Weight matrices Rk and Qk
    Rk = 0.1
    Qk = 5e4

    init_Pt = np.eye(4)  # Initial state covariance
    init_xtt_1 = 3e-2 * np.random.randn(4, 1)  # Initial state mean

elif(situation == "scenario 2"):
    # Weight matrices Rk and Qk
    Rk = 0.01
    Qk = 5e3

    # Initial state covariance and mean
    init_Pt = np.eye(4)
    init_xtt_1 = 1e-2 * np.random.randn(4, 1)


# Initialize yt
y0 = np.array([[0]])
next_yt = y0

# Simulation time parameters
tspan = [0, 20]
samp_time = 0.1

# init_Pt = np.eye(4)  # Initial state covariance
# init_xtt_1 = 3e-2 * np.random.randn(4, 1)  # Initial state mean

all_Ys = []
all_Us = []
all_covs = []

t_array = np.arange(tspan[0], tspan[1], samp_time)

# controller
controller = robust_mpc(A, B, C, Rk, Qk, Hu, Hp, 3e-2, 3e-2, init_Pt, init_xtt_1)

# Simulation loop
for e, t in enumerate(t_array):

    # collect new data
    yt = next_yt

    # Calculate optimal control input
    xtt, utt = controller.step(yt, rt)

    # Apply utt to the system
    if(situation == "scenario 1"):
        vtt = np.random.randn(4, 1)
        next_xtt = np.matmul(A, xtt) + (B * utt) + np.matmul(controller.G1, vtt)
        next_yt = np.matmul(C, next_xtt) + np.matmul(controller.D1, vtt)
    elif(situation == "scenario 2"):
        next_xtt = xtt + (servo_system.dt * servo_system.non_lin_dyn(xtt, utt, e + 1))
        next_yt = np.matmul(C, next_xtt)

    # Store variables for plotting
    all_Us.append(utt[0, 0])
    all_Ys.append(yt[0, 0])
    all_covs.append(controller.Vt[0, 0])


plt.figure()
plt.plot(t_array, all_Ys, label='Output')
plt.axhline(y=r, color='k', linestyle='--', label='Set-point= (pi / 2)')
plt.fill_between(t_array, y1=[y + np.sqrt(c) for y, c in zip(all_Ys, all_covs)], y2=[y - np.sqrt(c) for y, c in zip(all_Ys, all_covs)], alpha=0.5)
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

# Saving all the arrays
# np.save("npy_files/exp1/rob_mpc_lin_dyn_y", np.array(all_Ys))
# np.save("npy_files/exp1/rob_mpc_lin_dyn_u", np.array(all_Us))
# np.save("npy_files/exp1/rob_mpc_lin_dyn_cov", np.array(all_covs))