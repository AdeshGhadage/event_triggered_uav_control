import numpy as np

class FDICompensator:
    def __init__(self, num_rotors=6, threshold=0.1):
        """
        Initialize the Fault Detection and Isolation (FDI) compensator.
        
        Parameters:
            num_rotors (int): Number of rotors.
            threshold (float): Fault detection threshold.
        """
        self.num_rotors = num_rotors
        self.threshold = threshold
        self.rotor_states = np.ones(num_rotors)  # 1 for healthy, 0 for faulty
        
    def detect_faults(self, rotor_speeds, expected_speeds):
        """
        Detect faulty rotors by comparing actual and expected speeds.
        
        Parameters:
            rotor_speeds (numpy.ndarray): Actual rotor speeds.
            expected_speeds (numpy.ndarray): Expected rotor speeds.
            
        Returns:
            numpy.ndarray: Updated rotor states (1 for healthy, 0 for faulty).
        """
        error = np.abs(rotor_speeds - expected_speeds)
        self.rotor_states = (error < self.threshold).astype(float)
        return self.rotor_states
    
    def compensate_thrust(self, desired_thrust, rotor_states):
        """
        Compensate for faulty rotors by adjusting thrust allocation.
        
        Parameters:
            desired_thrust (float): Desired total thrust.
            rotor_states (numpy.ndarray): Current rotor states.
            
        Returns:
            numpy.ndarray: Compensated rotor speeds.
        """
        healthy_rotors = np.sum(rotor_states)
        if healthy_rotors == 0:
            return np.zeros(self.num_rotors)
        
        # Redistribute thrust among healthy rotors
        base_thrust = desired_thrust / healthy_rotors
        compensated_speeds = np.where(rotor_states == 1, base_thrust, 0)
        
        return compensated_speeds