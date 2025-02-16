import numpy as np

# A disturbance observer that:

# Estimates external disturbances based on the difference between expected and actual acceleration
# Uses a gain matrix for tuning the observer response
# Provides continuous updates to the disturbance estimate

class DisturbanceObserver:
    def __init__(self, mass, gain_matrix=None):
        """
        Initialize the disturbance observer.
        
        Parameters:
            mass (float): UAV mass.
            gain_matrix (numpy.ndarray): Observer gain matrix (3x3).
        """

        self.M = mass
        if gain_matrix is None:
            self.gain_matrix = np.eye(3)
        else:
            self.gain_matrix = gain_matrix

        self.estimated_disturbance = np.zeros(3)

    def update(self, state, thrust_direction, total_thrust, dt):
        """
        Update the disturbance estimate.
        
        Parameters:
            state (numpy.ndarray): Current UAV state [x, y, z, vx, vy, vz].
            thrust_direction (numpy.ndarray): Current thrust direction vector.
            total_thrust (float): Current total thrust.
            dt (float): Time step.
            
        Returns:
            numpy.ndarray: Updated disturbance estimate.
        """

        velocity = state[3:]
        
        # Nominal acceleration (without disturbance)
        nominal_accel = (total_thrust / self.M) * thrust_direction - np.array([0, 0, 9.81])
        
        # Actual acceleration (approximated from velocity)
        actual_accel = velocity / dt
        
        # Update disturbance estimate
        error = actual_accel - nominal_accel
        self.estimated_disturbance += dt * np.dot(self.gain_matrix, error)
        
        return self.estimated_disturbance
