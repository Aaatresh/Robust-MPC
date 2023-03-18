import numpy as np
from utils.utils import *
# import system_config as servo_system

# Model parameters
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

# del_t = 0.1
dt = 0.1

# Define bound on input voltage in volts
Vmax = 220

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

C = np.array([
    [1, 0, 0, 0]
])


def non_lin_dyn(x, u, k):

    """
        Simulate discrete time non-linear dynamics. Find next state given current state, input and time step number.
    """

    # Model parameters
    eps_max = 0
    eps_min = 0

    L = 0.8  # armature coil inductance
    J_m = 0.5 * (1 + eps_max)  # motor inertia
    beta_m = 0.1 * (1 + eps_max)  # motor viscous friction coefficient
    R = 20 * (1 + eps_min)  # resistance of armature
    Kt = 10 * (1 + eps_max)  # motor constant
    rho = 20 * (1 + eps_min)  # gear ratio
    k_theta = 1280.2 * (1 + eps_min)  # torsional rigidity
    J_l = 25 * (1 - eps_max)  # nominal load inertia
    beta_l = 25 * (1 + eps_max)  # load viscous friction coefficient
    alpha_l0, alpha_l1, alpha_l2 = [0.5, 10, 0.5]  # load non-linear friction parameters
    alpha_m0, alpha_m1, alpha_m2 = [0.1, 2, 0.5]  # motor non-linear friction params

    xdot = np.zeros_like(x)

    xdot[0, 0] = x[1, 0]

    # Evaluate non-linear coefficients
    Tfl = (alpha_l0 + (alpha_l1 * np.exp(-alpha_l2 * np.abs(x[1, 0])))) * sign(x[1, 0])
    Ts = (k_theta / rho) * ((x[2, 0] / rho) - x[0, 0])
    xdot[1, 0] = (1 / J_l) * ((rho * Ts) - (beta_l * x[1, 0]) - Tfl)

    xdot[2, 0] = x[3, 0]

    Tfm = (alpha_m0 + (alpha_m1 * np.exp(-alpha_m2 * np.abs(x[3, 0])))) * sign(x[3, 0])
    Im = ((u - (Kt * x[3, 0])) / R) * (1 - np.exp(-R * k * dt / L))
    Tm = Kt * Im
    xdot[3, 0] = (1 / J_m) * (Tm - Ts - (beta_m * x[3, 0]) - Tfm)

    return xdot
