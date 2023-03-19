from utils.utils import sign, load_yaml, cnt_to_dst
import numpy as np


class servo_mech_plant:

    """
    Class definition of a servo-mechanical system.
    """

    def __init__(self, model_params_filepath):

        """
        Constructor.

        Args:
            model_params_filepath - Filepath to servo mechanical system model parameters.
        """

        # Load all dynamics and model parameters from yaml file nad store in dictionary
        self.model_params = load_yaml(model_params_filepath)
        self.validate_params()


        self.cont_lin_state_space = None
        self.disc_lin_state_space = None

        self.init_lin_dyn()

    def validate_params(self):
        """
        Function to validate model parameters.
        """

        necessary_keys = ["dt", "Vmax"]
        model_types = ["linear_model", "nonlinear_model"]
        for key in necessary_keys + model_types:
            if(key not in self.model_params.keys()):
                raise KeyError(f"Key: [{key}] required for simulating this system.")
            elif(key in necessary_keys and type(self.model_params[key]) not in [float, int]):
                raise TypeError(f"Value of key: [{key}] must be either of type float or int.")
            elif(key in necessary_keys and self.model_params[key] < 0):
                raise ValueError(f"Key: [{key}] must be greater than or equal to 0.")

        model_keys = {
            "linear_model": ['L', 'J_m', 'beta_m', 'R', 'Kt', 'rho', 'k_theta', 'J_l', 'beta_l'],
            "nonlinear_model": ['eps_max', 'eps_min', 'L', 'J_m', 'beta_m', 'R', 'Kt', 'rho', 'k_theta',
                                'J_l', 'beta_l', 'alpha_l0', 'alpha_l1', 'alpha_l2', 'alpha_m0', 'alpha_m1', 'alpha_m2']
        }
        for model in model_types:
            for subkey in model_keys[model]:
                if(subkey not in self.model_params[model].keys()):
                    raise KeyError(f"Key: [{model}][{subkey}] required for simulating this system.")
                elif(type(self.model_params[model][subkey]) not in [float, int]):
                    raise TypeError(f"Value of key: [{model}][{subkey}] must be either of type float or int.")
                elif(self.model_params[model][subkey] < 0):
                    raise ValueError(f"Key: [{model}][{subkey}] must be greater than or equal to 0.")


    def init_lin_dyn(self):

        """
        Function to initialize linear dynamics continuous time state space model using a configuration file containing
        model parameters for a servo-mechanical system.

        Args:
            filepath - File path to linear dynamics parameters.

        Returns:
            Dictionary containing continuous time state space model matrices Ac, Bc, C

        """

        lin_params = self.model_params["linear_model"]

        # Define A
        Ac = np.array([
            [0, 1, 0, 0],
            [-(lin_params["k_theta"]) / lin_params["J_l"], -lin_params["beta_l"] / lin_params["J_l"],
                            (lin_params["k_theta"]) / (lin_params["rho"] * lin_params["J_l"]), 0],
            [0, 0, 0, 1],
            [(lin_params["k_theta"]) / (lin_params["rho"] * lin_params["J_m"]), 0,
                            (-lin_params["k_theta"]) / ((lin_params["rho"] ** 2) * lin_params["J_m"]),
                            - ((lin_params["Kt"] ** 2) / (lin_params["J_m"] * lin_params["R"])) - (lin_params["beta_m"] / lin_params["J_m"])]
        ])

        # Define B
        Bc = np.array([
            [0],
            [0],
            [0],
            [lin_params["Kt"] / (lin_params["J_m"] * lin_params["R"])]
        ])

        C = np.array([
            [1, 0, 0, 0]
        ])

        # Create a dictionary for a continuous time linear state space model
        self.cont_lin_state_space = {
            "Ac": Ac,
            "Bc": Bc,
            "C": C
        }

        self.disc_lin_state_space = cnt_to_dst(self.cont_lin_state_space, self.model_params["dt"])


    def nonlin_dyn_step(self, x, u, k):

        """
            Simulate discrete time non-linear dynamics. Find next state given current state, input and time step number.
        """

        nonlin_params = self.model_params["nonlinear_model"]

        xdot = np.zeros_like(x)

        xdot[0, 0] = x[1, 0]

        # Evaluate non-linear coefficients
        Tfl = (nonlin_params["alpha_l0"] + (nonlin_params["alpha_l1"] * np.exp(-nonlin_params["alpha_l2"] * np.abs(x[1, 0])))) * sign(x[1, 0])
        Ts = (nonlin_params["k_theta"] / nonlin_params["rho"]) * ((x[2, 0] / nonlin_params["rho"]) - x[0, 0])
        xdot[1, 0] = (1 / nonlin_params["J_l"]) * ((nonlin_params["rho"] * Ts) - (nonlin_params["beta_l"] * x[1, 0]) - Tfl)

        xdot[2, 0] = x[3, 0]

        Tfm = (nonlin_params["alpha_m0"] + (nonlin_params["alpha_m1"] * np.exp(-nonlin_params["alpha_m2"] * np.abs(x[3, 0])))) * sign(x[3, 0])
        Im = ((u - (nonlin_params["Kt"] * x[3, 0])) / nonlin_params["R"]) * (1 - np.exp(-nonlin_params["R"] * k * self.model_params["dt"] / nonlin_params["L"]))
        Tm = nonlin_params["Kt"] * Im
        xdot[3, 0] = (1 / nonlin_params["J_m"]) * (Tm - Ts - (nonlin_params["beta_m"] * x[3, 0]) - Tfm)

        return xdot

    def fwd_prop_lin_model(self, xtt, utt, G1, D1):
        """
            Forward propogate linear dynamics model by one time step given current state and control signal.

            Args:
                xtt - Current state
                utt - Control signal at this time step
                G1 - Standard deviation matrix associated with action model
                D1 - Standard deviation matrix associated with sensor model

            Returns:
                next_xtt - Next state
                next_yt - Next output
        """

        vtt = np.random.randn(self.disc_lin_state_space["A"].shape[0], 1)
        next_xtt = np.matmul(self.disc_lin_state_space["A"], xtt) + (self.disc_lin_state_space["B"] * utt) + np.matmul(G1, vtt)
        next_yt = np.matmul(self.disc_lin_state_space["C"], next_xtt) + np.matmul(D1, vtt)

        return next_xtt, next_yt

    def fwd_prop_nonlin_model(self, xtt, utt, e):
        """
            Forward propogate non-linear dynamics model by one time step given current state and control signal.

            Args:
                xtt - Current state
                utt - Control signal at this time step
                e - index number of time step
            Returns:
                next_xtt - Next state
                next_yt - Next output
        """
        next_xtt = xtt + (self.model_params["dt"] * self.nonlin_dyn_step(xtt, utt, e + 1))
        next_yt = np.matmul(self.disc_lin_state_space["C"], next_xtt)

        return next_xtt, next_yt
