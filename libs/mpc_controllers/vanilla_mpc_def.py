"""
    Module with vanilla MPC definition.
"""

import numpy as np


class abstract_controller:
    """
        Abstract controller class.
    """

    def __int__(self):
        self.init_controller()
        pass

    def init_controller(self):
        pass

    def get_utt(self):
        pass


class vanilla_mpc(abstract_controller):
    """ Class definition of a standard MPC controller """

    def __init__(self, disc_lin_state_space, config_params, Rk, Qk, init_Pt, init_xtt_1, control_bounds=None):
        """
            Constructor

            Args:
                disc_lin_state_space - Discretized linear state space model with key-value pairs:
                    A - Discrete time 'A' state space matrix.
                    B - Discrete time 'B' state space matrix.
                    C - Discrete time 'C' state space matrix.

                controller_config_params - Dictionary representing configuration parameters for this controller

                Rk - Control input gain on diagonal.
                Qk - Output gain on diagonal.

                init_Pt - Initialization of state estimate covariance matrix.
                init_xtt_1 - Initialization of state estimate mean.

                control_bounds - Default value = None. This represents bounds to be applied on the control signal and
                    clamp it to this. When provided as a list, the first element represents the lower bound and the
                    second represents the upper bound.

            Returns:
                -
        """

        self.A = disc_lin_state_space["A"]
        self.B = disc_lin_state_space["B"]
        self.C = disc_lin_state_space["C"]

        self.Hu = config_params["Hu"]
        self.Hp = config_params["Hp"]

        self.Qk_mat = None
        self.Rk_mat = None

        self.psi = None             # psi matrix
        self.gamma = None           # gamma matrix

        # initialize controller
        self.init_controller(Rk, Qk)

        # Standard deviation of state and measurement noise
        self.G1 = config_params["act_model_std"] * np.diag(np.array([0, 1, 1, 1]))
        self.D1 = config_params["sen_model_std"] * np.array([[1, 0, 0, 0]])

        # Initial state covariance and mean
        self.Pt = init_Pt

        # state of the system
        self.xtt = init_xtt_1

        # Store control signal lower and upper bounds. If not None, store in index 0 and index 1 respectively
        self.control_bounds = control_bounds


    def init_controller(self, Rk, Qk):
        """
            Function to initialize the standard MPC.

            Args:
                Rk - Diagonal elementwise control input gain
                Qk - Diagonal elementwise prediction gain

            Returns:
                -
        """

        # Weight matrices Rk and Qk
        Rk_diag = Rk * np.ones((self.Hu,))
        self.Rk_mat = np.diag(Rk_diag)
        Qk_diag = Qk * np.ones((self.Hp,))
        self.Qk_mat = np.diag(Qk_diag)

        # Making psi
        psi_T = np.matmul(self.C, self.A).T
        for c in range(2, self.Hp + 1):
            psi_T = np.hstack((psi_T, np.matmul(self.C, np.linalg.matrix_power(self.A, c)).T))

        self.psi = psi_T.T

        # Making gamma
        gamma = 1 * np.ones((self.Hp, self.Hu))
        for col in range(self.Hu):

            col_data = []

            for z in range(col):
                col_data.append(np.array([[0]]))

            for row in range(self.Hp - col):
                col_data.append(np.matmul(self.C, np.matmul(np.linalg.matrix_power(self.A, row), self.B)))

            gamma[:, col] = np.array(col_data).squeeze()

        self.gamma = gamma


    def get_utt(self, xtt, rt):
        """
           Solve the unconstrained MPC problem through a closed form expression to find the control input
           given the state estimate at that time step.

           Args:
               xtt - predicted state
               rt - set point for the next Hp timesteps

           Returns:
               utt - Optimal control
        """

        # This is used to extract the first time step's control
        W = np.array([
            [1, 0, 0]
        ])

        # Calculate intermediate terms in the computation of the optimal control input
        term1 = np.matmul(self.gamma.T, np.matmul(self.Qk_mat, self.gamma)) + self.Rk_mat
        term2 = np.matmul(self.gamma.T, np.matmul(self.Qk_mat, (rt - np.matmul(self.psi, xtt))))

        # Find optimal control input
        utt = np.matmul(W, np.matmul(np.linalg.inv(term1), term2))

        # Clamp calculated control input to bounds if not None
        if(self.control_bounds is not None):
            if ((utt > self.control_bounds[1])):
                utt = np.array([[self.control_bounds[1]]])
            elif (utt < self.control_bounds[0]):
                utt = np.array([[self.control_bounds[0]]])

        return utt


    def state_prediction(self, yt):
        """
            Function that performs the state prediction step in a standard Kalman filter.

            Args:
                yt - output of the system

            Returns:
                xtt - Predicted state of the system.
        """

        # Find Lt
        term1 = np.matmul(self.Pt, self.C.T)
        term2 = np.linalg.pinv(np.matmul(self.C, np.matmul(self.Pt, self.C.T)) + (np.matmul(self.D1, self.D1.T)))
        self.Lt = np.matmul(term1, term2)

        # Prediction step to find xtt
        xtt = self.xtt + (self.Lt * (yt - np.matmul(self.C, self.xtt)))

        return xtt


    def state_update(self, utt, yt):

        """
            Function that performs the state update step.

            Args:
                utt - Optimal control input.
                yt - Output of the system

            Returns:
                -
        """

        # Kalman gain
        kg = np.matmul(self.A, self.Lt)

        # Update pt
        term1 = np.matmul(self.A, np.matmul(self.Pt, self.A.T))
        term2 = np.matmul(self.C, np.matmul(self.Pt, self.C.T)) + np.matmul(self.D1, self.D1.T)
        term3 = np.matmul(kg, np.matmul(term2, kg.T))
        self.Pt = term1 - term3 + np.matmul(self.G1, self.G1.T)

        # Prediction step
        self.xtt = np.matmul(self.A, self.xtt) + (kg * (yt - np.matmul(self.C, self.xtt))) + (self.B * utt)

    def step(self, yt, rt):

        """
            Function to perform a step of the standard MPC optimal control computation.

            Args:
                yt - Output of the system.
                rt - Set points for the next Hp time steps.

            Returns:
                xtt - Predicted state estimate
                utt - Calculated optimal control
        """

        xtt = self.state_prediction(yt)             # State prediction
        utt = self.get_utt(xtt, rt)                 # Calculate optimal control
        self.state_update(utt, yt)                  # State update

        return xtt, utt