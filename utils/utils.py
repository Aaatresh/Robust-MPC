import numpy as np
import scipy.linalg
import os
import matplotlib.pyplot as plt


def cnt_to_dst(cont_lin_state_space, dt):

    """
        Function to convert a continuous time state space model to a discrete time state space model.
            x_dot = Ac(x) + Bc(u)

        Args:
            cont_lin_state_space - Dictionary containing the following as keys
                'Ac': Ac - Continuous time 'A' matrix.
                'Bc': Bc - Continous time 'B' matrix.
                'C': C - 'C' matrix

            dt - sampling time for discretization.

        Returns:
            Ad - Discrete time 'A' matrix
            Bd - Discrete time 'B' matrix
            C - 'C' matrix
    """

    Ad = scipy.linalg.expm(cont_lin_state_space["Ac"] * dt)
    Bd = np.matmul(np.linalg.pinv(cont_lin_state_space["Ac"]),
                   np.matmul((scipy.linalg.expm(cont_lin_state_space["Ac"] * dt) - np.eye(cont_lin_state_space["Ac"].shape[0],
                                                                                     cont_lin_state_space["Ac"].shape[1])), cont_lin_state_space["Bc"]))

    disc_lin_state_space = {
        "A": Ad,
        "B": Bd,
        "C": cont_lin_state_space["C"]
    }

    return disc_lin_state_space


def sign(x):

    """
        Signum function.
    """
    if(x > 0):
        return 1
    elif(x < 0):
        return -1
    else:
        return 0


def eval_kld(Pt, param_t, kld_thresh):
    """
        Evaluate KL divergence given the state's covariance matrix, inverse of the lagrange multiplier and
        the radius of a ball around the nominal model.
    """

    term1 = np.log(np.linalg.det(np.eye(4, 4) - (param_t * Pt)))
    term2 = np.trace(np.linalg.inv(np.eye(4, 4) - (param_t * Pt)) - np.eye(4, 4))

    term_total = term1 + term2 - kld_thresh

    return term_total


def bijection_algo(Pt, kld_thresh):

    """ Root finding algorithm - Bijection algorithm, given the state's covariance matrix """

    eps = 1e-7
    param1 = eps

    eigs = np.linalg.eigvals(Pt + 1e-7 * np.eye(4, 4))
    eigs = eigs.real

    lam = np.max(np.abs(eigs))

    if(((1 / lam) > 2 * eps) and (lam > 1e-5)):
        param2 = (1 / lam) - eps
    else:
        param2 = eps

    param_t = param1

    while(np.abs(param1 - param2) > eps):

        param_t = (param1 + param2) / 2
        kld = eval_kld(Pt, param_t, kld_thresh)

        if(kld < 0):
            param1 = param_t
        else:
            param2 = param_t

    return param_t


def load_yaml(filepath):
    """
    Function to open yaml file and get data from it.

    Args:
        filepath - Relative file path to yaml file.

    Returns:
        Dictionary containing data stored in yaml file.
    """
    # Import pyyaml
    import yaml

    # Open and read yaml file data
    with open(filepath, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exception:
            raise exception

    return data


def visualize_controller(collected_data, args, CONTROLLER_NAME, setpoint):

    t_array, all_Ys, all_Us, all_covs = collected_data

    plt.figure()
    plt.plot(t_array, all_Ys, label='Output')
    plt.axhline(y=setpoint, color='k', linestyle='--', label='Set-point= (pi / 2)')
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