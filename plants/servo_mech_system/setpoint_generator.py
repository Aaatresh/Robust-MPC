import numpy as np


def const_setpoint_gen(const_value=0, Hp=5):
    """
    Function that generates a constant valued setpoint signal for a given horizon

    Args:
        const_value - Value of amplitude of constant valued signal.
        Hp - Number of future timesteps or the length of the prediction horizon

    Returns:
        Constant valued setpoint signal
    """

    # Find setpoint signal
    sp_signal = const_value * np.ones((Hp, 1))

    return sp_signal