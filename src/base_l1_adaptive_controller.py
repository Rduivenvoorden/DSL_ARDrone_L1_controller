#!/usr/bin/env python2

""" L1 adaptive controller implementation.

Rikky Duivenvoorden

This implements an L1 adaptive controller based on:
N. Hovakimyan and C. Cao, Adaptive Control Theory: Guaranteed Robustness with
    Fast Adaptation. Philadelphia, PA: Society for Industrial and Applied
    Mathematics, 2010.

"""

import numpy as np

__all__ = ['L1AdaptiveController', 'PIDController', 'LTISystem']


class L1AdaptiveController(object):
    """L1 adaptive controller implementation."""

    def __init__():
        """Initialize the L1 adaptive controller."""
        
        # Low pass filter
        self.LPF = LTISystem(state=np.zeros(1, dtype=np.float)),
                             state_matrix=np.array(self.m),
                             input_matrix=np.array(-self.m),
                             output_matrix=np.array(1, dtype=np.float)))
        
        # State or output predictor
        self.state_predictor = LTISystem(state=np.zeros(1, dtype=np.float),
                                         state_matrix=np.array(self.m),
                                         input_matrix=np.array(-self.m),
                                         output_matrix=np.array(1,
                                         dtype=np.float))
        # L1 Adaptive Gain
        self.Gamma = 5000.
        self.sigma_hat = None # TODO: check dim
        self.epsilon_sigma = None # TODO: check dim
        self.sigma_hat_max = None # TODO: check dim
        self.x_ref = None # TODO: check dim

    def reinitialize_l1_states(self):
        """ Reinitialize L1 states."""

        self.LPF.reinitialize_system()
        self.state_predictor.reinitialize_system()

        self.sigma_hat = np.zeros(self.sigma_hat.shape[0], dtype=np.float)
        self.x_ref = np.zeros(self.x_ref.shape[0], dtype=np.float)

    def get_control_input(self, desired_state, actual_state, dt):
        """Calculate the L1 control command based on error and time difference.
        
        Parameters
        ----------
        desired_state
            The desired state of the system, which the L1 adaptive controller
            aims to achieve based on the actual state currently.
        actual_state
            The output of the system, which the L1 adaptive controller is to
            control.
        dt: float
            The time difference between the current control command computation
            and the previous call to get_control_input().
        
        """

        # Calculate y-tilde - error between actual and reference state position
        y_tilde = self.x_ref - actual_state

        if self.use_projection_operator:
            # Projection Based Adaptation

            # Use projection operator to update sigma_hat based on y_tilde
            f = (((self.epsilon_sigma + 1) * np.sum(self.sigma_hat ** 2) -
                 self.sigma_hat_max ** 2) /
                 (self.epsilon_sigma * self.sigma_hat_max ** 2))

            grad_f = (2 * (self.epsilon_sigma + 1) /
                      (self.epsilon_sigma * self.sigma_hat_max ** 2) *
                      self.sigma_hat)

            projection_result = -y_tilde
            if f >= 0 and grad_f.dot(y_tilde) <= 0:
                projection_result += (grad_f / np.linalg.norm(grad_f) *
                                      grad_f.dot(y_tilde) * f)

            # Multiply by adaptive Gain and integrate
            sigma = self.sigma_hat + dt * self.Gamma * projection_result
            sigma_max = self.sigma_hat_max * (1 + self.epsilon_sigma)
            
            # Keep track of how often sigma hat reaches its maximum
            if np.any(sigma > sigma_max):
                self._sigma_clip_counter += 1
                if (self._sigma_clip_counter > self._sigma_clip_limit and 
                    self._sigma_clip_limit is not None):
                    print("Reinitializing L1: sigma clipped too often")
                    self.reinitialize_l1_states()                    
                    # Restart counter
                    self._sigma_clamp_counter = 0

            self.sigma_hat = np.clip(sigma, None, sigma_max)

        else:
            # Piecewise Constant Adaptation

            # Piecewise constant update of sigma_hat based on y_tilde
            # TODO: Implementation currently assumes A_m diagonal, use
            # scipy.linalg expm for non-diagonal A_m

            Ad = np.diag(self.state_predictor.A)
            Bd = np.diag(self.state_predictor.B)

            self.sigma_hat = (-Ad * np.exp(Ad * dt) * y_tilde /
                              (Bd * (np.exp(Ad * dt) - 1)))

        # Find L1 desired by low-pass filtering tracking error
        track_error = self.L1_input - self.sigma_hat

        # Pass through the low pass filter to get the control command
        x_L1_des = self.LPF.get_next_state(track_error, dt)

        # Update the state of the state predictor
        predictor_input = x_L1_des + self.sigma_hat
        self.x_ref = self.state_predictor.get_next_state(predictor_input, dt)
        
        # Return the L1 control command 
        return x_L1_des


class LTISystem(object):
    """Single-input single-output linear time-invariant system.
    
    Parameters
    ----------
    state : ndarray
        Initial state, x_0, of the system. The length of initial state array
        determines the order of the SISO LTI system.
    state_matrix : ndarray
        State matrix, A, of the system with two dimensions of the same length
        equal to the length of the state array, or a scalar float for 1D
    input_matrix : ndarray
        Input matrix, B, of the system in the form of an array equal in length
        to the state array.
    output_matrix : ndarray
        Output matrix, C, of the system in the form of an array equal in length
        to the state array.
    feedforward : float, optional
        Feedforward term, D, 
    """
    
    def __init__(state=np.zeros(2, dtype=np.float),
                 state_matrix=np.zeros((2, 2), dtype=np.float),
                 input_matrix=np.zeros(2, dtype=np.float)
                 output_matrix=np.zeros(2, dtype=np.float)
                 feedforward=0.0):
        """Initialize the system."""
        self.x = state
        self.A = state_matrix
        self.B = input_matrix
        self.C = output_matrix
        self.D = feedforward

        if (self.x.__class__ == self.A.__class__ == self.B.__class__ == 
            np.ndarray):
            if not self.x.ndim == 1:
                raise RuntimeError('State ndim ERROR: shape (n,) is expected.')
            if not self.A.ndim == 2:
                raise RuntimeError('State matrix ndim ERROR: (n, n) expected.')
            if not self.B.ndim == 1:
                raise RuntimeError('State ndim ERROR: shape (n,) is expected.')
            if not self.C.ndim == 2:
                raise RuntimeError('State matrix ndim ERROR: (n, n) expected.')
        else
            raise RuntimeError('numpy.ndarray expected for x, A, B and C.')
    
    def reinitialize_system(self):
        """Resets the system to state zero."""
        
        self.x = np.zeros(self.x.shape[0])

    def get_next_state(self, system_input, dt)
        """Returns the next state

        Parameters
        ----------
        system_input : float
            Initial state, x_0, of the system. The length of initial state 
            array determines the order of the SISO LTI system.
        dt : float
            State matrix, A, of the system
        
        Returns
        -------
        system_output : float
            The output of the system dt seconds after the last time
            get_next_state() was called.
            
        """
        
        # Compute the state change based on input and current state
        state_change = self.A.dot(self.x) + self.B.dot(system_input)
        
        # Assume small step size and integrate using Euler's Method
        # TODO: change to more advanced numerical integration 
        self.x += dt*state_change
        
        # Compute output
        output = self.output_matrix.dot(self.x) + self.D*system_input
        
        # Return the output
        return output


class PIDController(object):
    """PID controller implementation."""

    def __init__():
        """Initialize the controller."""
        pass
    
    def get_control_input():
        """Calculate the control input based on error and time difference."""
        pass
