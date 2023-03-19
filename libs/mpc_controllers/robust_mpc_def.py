"""
    Module with robust MPC definition.
"""

from utils.utils import *
from libs.mpc_controllers.vanilla_mpc_def import vanilla_mpc


class robust_mpc(vanilla_mpc):
    """ Class definition of a robust MPC controller and inheriting from the vanilla MPC controller """

    def __init__(self, disc_lin_state_space, controller_config_params, Rk, Qk, init_Pt, init_xtt_1, control_bounds=None):
        """
            Constructor

            Args:
                A - Discrete time 'A' state space matrix.
                B - Discrete time 'B' state space matrix.
                C - Discrete time 'C' state space matrix.

                Rk - Control input gain on diagonal.
                Qk - Output gain on diagonal.

                Hu - Prediction horizon associated with control input.
                Hp - Prediction horizon associated with output.

                act_model_std - Standard deviation of noise associated with action model.
                act_model_std - Standard deviation of noise associated with sensor model.

                init_Pt - Initialization of state estimate covariance matrix.
                init_xtt_1 - Initialization of state estimate mean.

            Returns:
                -
        """

        super().__init__(disc_lin_state_space, controller_config_params, Rk, Qk, init_Pt, init_xtt_1,
                         control_bounds)

        # store KLD threshold
        self.kld_thresh = controller_config_params["kld_thresh"]


    def state_prediction(self, yt):
        """
            Function that performs the state prediction step in a robust Kalman filter.

            Args:
                yt - output of the system

            Returns:
                xtt - Predicted state of the system.
        """

        # Find param_t using the bijection algo
        param_t = bijection_algo(self.Pt, self.kld_thresh)

        # Determine Vt
        self.Vt = np.linalg.pinv(np.linalg.pinv(self.Pt) - (param_t * np.eye(4, 4)))

        # Find Lt
        term1 = np.matmul(self.Vt, self.C.T)
        term2 = np.linalg.pinv(np.matmul(self.C, np.matmul(self.Vt, self.C.T)) + np.matmul(self.D1, self.D1.T))
        self.Lt = np.matmul(term1, term2)

        # Prediction step to find xtt
        xtt = self.xtt + (self.Lt * (yt - np.matmul(self.C, self.xtt)))

        return xtt


    def state_update(self, utt, yt):
        """
            Function that performs the state update step in a robust Kalman filter.

            Args:
                utt - Optimal control input.
                yt - Output of the system

            Returns:
                -
        """

        # Kalman gain
        kg = np.matmul(self.A, self.Lt)

        # Update pt
        term1 = np.matmul(self.A, np.matmul(self.Vt, self.A.T))
        term2 = np.matmul(self.C, np.matmul(self.Vt, self.C.T)) + np.matmul(self.D1, self.D1.T)
        term3 = np.matmul(kg, np.matmul(term2, kg.T))
        self.Pt = term1 - term3 + np.matmul(self.G1, self.G1.T)

        # Prediction step
        self.xtt = np.matmul(self.A, self.xtt) + (kg * (yt - np.matmul(self.C, self.xtt))) + (self.B * utt)