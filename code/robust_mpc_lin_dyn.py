"""
Performance of the robust MPC when the approximated model and the actual model coincide
"""

# Importing necessary libraries
import matplotlib.pyplot as plt
from controller_config import *
import system_config as servo_system
from utils import *
from vanilla_mpc_def import vanilla_mpc

# Convert the model to discrete-time
A, B, C = cnt_to_dst(servo_system.Ac, servo_system.Bc, servo_system.C, servo_system.dt)

# Set point
r = np.pi / 2
rt = r * np.ones((Hp, 1))

# Weight matrices Rk and Qk
Rk = 0.1
Rk_diag = Rk * np.ones((Hu,))
Rk_mat = np.diag(Rk_diag)

# Qk = 5e4
Qk = 5e4
Qk_diag = Qk * np.ones((Hp,))
Qk_mat = np.diag(Qk_diag)

# This is used to extract the first time step's control
W = np.array([
    [1, 0, 0]
])

# Making psi
psi_T = np.matmul(C, A).T
for c in range(2, Hp+1):
    psi_T = np.hstack((psi_T, np.matmul(C, np.linalg.matrix_power(A, c)).T))

psi = psi_T.T

# Making gamma
gamma = 1 * np.ones((Hp, Hu))
for col in range(Hu):

    col_data = []

    for z in range(col):
        col_data.append(np.array([[0]]))

    for row in range(Hp-col):
        col_data.append(np.matmul(C, np.matmul(np.linalg.matrix_power(A, row), B)))

    gamma[:, col] = np.array(col_data).squeeze()


def get_utt(xtt):
    """
       Solve the unconstrained MPC problem through a closed form expression to find the control input
       given the state estimate at that time step.
    """

    term1 = np.matmul(gamma.T, np.matmul(Qk_mat, gamma)) + Rk_mat
    term2 = np.matmul(gamma.T, np.matmul(Qk_mat, (rt - np.matmul(psi, xtt))))

    utt = np.matmul(W, np.matmul(np.linalg.inv(term1), term2))

    return utt


# Initialize yt
y0 = np.array([[0]])
next_yt = y0

# Simulation time parameters
tspan = [0, 20]
samp_time = 0.1

Pt = np.eye(4)  # Initial state covariance
Vt = np.eye(4)
xtt_1 = 3e-2 * np.random.randn(4, 1)  # Initial state mean

all_Ys = []
all_Us = []
all_covs = []

# Standard deviation of state and measurement noise
G1 = 3e-2 * np.diag(np.array([0, 1, 1, 1]))
D1 = 3e-2 * np.array([[1, 0, 0, 0]])

t_array = np.arange(tspan[0], tspan[1], samp_time)

# Simulation loop
for t in t_array:

    # collect new data
    yt = next_yt

    # Find param_t using the bijection algo
    param_t = bijection_algo(Pt)

    # Determine Vt
    Vt = np.linalg.pinv(np.linalg.pinv(Pt) - (param_t * np.eye(4, 4)))

    # Find Lt
    term1 = np.matmul(Vt, C.T)
    term2 = np.linalg.pinv(np.matmul(C, np.matmul(Vt, C.T)) + np.matmul(D1, D1.T))
    Lt = np.matmul(term1, term2)

    # Prediction step to find xtt
    xtt = xtt_1 + (Lt * (yt - np.matmul(C, xtt_1)))

    # Get control
    utt = get_utt(xtt)

    # temp
    if (utt > 220):
        utt = np.array([[220]])
    elif (utt < -220):
        utt = np.array([[-220]])

    all_Us.append(utt[0, 0])

    # Apply utt to the system
    vtt = np.random.randn(4, 1)
    next_xtt = np.matmul(A, xtt) + (B * utt) + np.matmul(G1, vtt)
    next_yt = np.matmul(C, next_xtt) + np.matmul(D1, vtt)

    # Kalman gain
    kg = np.matmul(A, Lt)

    # Update pt
    term1 = np.matmul(A, np.matmul(Vt, A.T))
    term2 = np.matmul(C, np.matmul(Vt, C.T)) + np.matmul(D1, D1.T)
    term3 = np.matmul(kg, np.matmul(term2, kg.T))
    Pt = term1 - term3 + np.matmul(G1, G1.T)

    # Prediction step
    xtt_1 = np.matmul(A, xtt_1) + (kg * (yt - np.matmul(C, xtt_1))) + (B * utt)

    all_Us.append(utt[0, 0])
    all_Ys.append(yt[0, 0])
    all_covs.append(Vt[0, 0])



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