import numpy as np
import scipy.linalg
from code.controllers.controller_config import kld_thresh


def cnt_to_dst(Ac, Bc, C, dt):

    """
        Function to convert a continuous time state space model to a discrete time state space model.
            x_dot = Ac(x) + Bc(u)

        Args:
            Ac - Continuous time 'A' matrix.
            Bc - Continous time 'B' matrix.
            dt - sampling time for discretization.

        Returns:
            Ad - Discrete time 'A' matrix
            Bd - Discrete time 'B' matrix
    """

    Ad = scipy.linalg.expm(Ac * dt)
    Bd = np.matmul(np.linalg.pinv(Ac), np.matmul((scipy.linalg.expm(Ac * dt) - np.eye(Ac.shape[0], Ac.shape[1])), Bc))

    return Ad, Bd, C


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

    # term1 = -np.log(np.linalg.det(np.linalg.inv(np.eye(2, 2) - (param_t * Pt))))
    term1 = np.log(np.linalg.det(np.eye(4, 4) - (param_t * Pt)))
    term2 = np.trace(np.linalg.inv(np.eye(4, 4) - (param_t * Pt)) - np.eye(4, 4))

    term_total = term1 + term2 - kld_thresh

    return term_total


def bijection_algo(Pt):

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