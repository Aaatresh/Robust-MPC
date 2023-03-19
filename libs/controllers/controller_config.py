# prediction and control horizons
Hp = 10
Hu = 3

# Threshold for KL divergence. In the reference literature, this is mentioned as 'c'
kld_thresh = 0.1

# Define action model noise standard of deviation
act_model_std = 3e-2

# Define action model noise standard of deviation
sen_model_std = 3e-2


# Define elements for weight matrices for two different scenarios

def get_controller_weights(controller_name, scenario):
    """
    Function to obtain controller weights depending on the scenario under consideration

    Args:
        controller_name - Name of MPC controller, vanilla or robust
        scenario - Scenario under consideration. Can be either 1 or 2.

    Returns:
        Controller weights
    """
    if(controller_name == "robust"):
        if(scenario == 1):
            Rk = 0.1
            Qk = 5e4
        else:
            Rk = 0.01
            Qk = 5e3
    else:
        if(scenario == 1):
            Rk = 0.1
            Qk = 1e4
        else:
            Rk = 0.01
            Qk = 5e3

    return Rk, Qk
