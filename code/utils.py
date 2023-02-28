import numpy as np
import scipy.linalg

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

