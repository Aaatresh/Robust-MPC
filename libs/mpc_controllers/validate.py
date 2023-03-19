import warnings

necessary_keys = ['Hp', 'Hu', 'kld_thresh', 'act_model_std', 'sen_model_std']
necessary_keys_controllers = ["robust", "vanilla"]
necessary_sub_keys = [1, 2]
necessary_sub_sub_keys = ['Rk', 'Qk', 'init_Pt_std', 'init_xtt_std']


def check_file_format(control_config_params):
    """
        Validate file format through its dictionary.
    """

    # Check if keys are in dictionary
    for key in necessary_keys:
        if (key not in control_config_params.keys() and key not in necessary_keys_controllers):
            raise KeyError(f"Key: {key} not found in YAML file.")

    controller_keys_found = []
    keys_not_found = []
    for controller_key in necessary_keys_controllers:
        if (controller_key in control_config_params.keys()):
            controller_keys_found.append(controller_key)
        else:
            keys_not_found.append(controller_key)

    if ((len(necessary_keys_controllers) - len(controller_keys_found)) > 1):
        raise KeyError("At least one MPC controller name required.")
    elif ((len(necessary_keys_controllers) - len(controller_keys_found)) == 1):
        warnings.warn(f"Key: {keys_not_found[0]}, not found. Please make sure that this controller is not used until "
                      f"relevant configuration parameters are provided in the YAML controller configuration file.")

    for key in controller_keys_found:
        if (list(control_config_params[key].keys()) != necessary_sub_keys):
            raise ValueError(
                f"All sub-keys: {necessary_sub_keys} not found with key: {key} in YAML controller configuration file.")
        else:
            for subkey in necessary_sub_keys:
                if (list(control_config_params[key][subkey].keys()) != necessary_sub_sub_keys):
                    raise ValueError(
                        f"All subsub-keys: {necessary_sub_sub_keys} not found with key: {key}, and sub-key {subkey}, in "
                        f"YAML controller configuration file.")


def check_param_values(control_config_params):

    for key in necessary_keys:
        if(not type(control_config_params[key]) in [float, int]):
            raise TypeError(f"Data of key: {key} must be either float or int.")
        elif(control_config_params[key] <= 0):
            raise ValueError(f"Value of key: [{key}] must be strictly positive.")


    for key in necessary_keys_controllers:
        if(key in control_config_params.keys()):
            for subkey in necessary_sub_keys:
                for subsubkey in necessary_sub_sub_keys:
                    if(not type(control_config_params[key][subkey][subsubkey]) in [float, int]):
                        raise TypeError(f"Data of key: [{key}][{subkey}][{subsubkey}] must be either float or int.")
                    elif (control_config_params[key][subkey][subsubkey] <= 0):
                        raise ValueError(f"Value of key: [{key}][{subkey}][{subsubkey}] must be strictly positive.")


def validate_control_params(control_config_params):
    """
    Function to validate data read from a YAML file containing MPC controller configurations
    """

    check_file_format(control_config_params)

    check_param_values(control_config_params)

    return control_config_params


if __name__ == "__main__":

    from utils.utils import load_yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str)
    args = parser.parse_args()

    params = load_yaml(args.f)

    print(validate_control_params(params))