import numpy as np

class SixRotorUAV:
    def __init__(self, mass=1.0, g=9.81, air_resistance=(0.1, 0.1, 0.1), p_const=1.0):
        """
        Initialize the six-rotor UAV model.
        
        Parameters:
            mass (float): UAV mass (M).
            g (float): Gravitational acceleration.
            air_resistance (tuple): (r1, r2, r3) coefficients for the diagonal aerodynamic matrix Gamma.
            p_const (float): Constant p̆ used in total thrust calculation.
        """

        self.M = mass
        self.G = g
        self.Gamma = np.diag(air_resistance)
        self.p_const = p_const

        # UAV state vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # Euler angles (in radians): [roll (χ), pitch (ψ), yaw (φ)]
        self.euler = np.zeros(3)
    
    def set_euler_angles(self, roll, pitch, yaw):
        """
        Set the Euler angles of the UAV.
        
        Parameters:
            roll (float): Roll angle χ.
            pitch (float): Pitch angle ψ.
            yaw (float): Yaw angle φ.
        """
        self.euler = np.array([roll, pitch, yaw])
    
    def set_state(self, x, y, z, vx, vy, vz):
        """
        Set the UAV state vector.
        
        Parameters:
            x (float): x-coordinate.
            y (float): y-coordinate.
            z (float): z-coordinate.
            vx (float): x-velocity.
            vy (float): y-velocity.
            vz (float): z-velocity.
        """
        self.state = np.array([x, y, z, vx, vy, vz])

    def rotation_matrix(self, roll, pitch, yaw):
        """
        Compute the rotation matrix R from the body-fixed frame to the earth-fixed inertial frame.
        We use the ZYX Euler angle convention (yaw-pitch-roll).
        
        Parameters:
            roll (float): Roll angle (χ) in radians.
            pitch (float): Pitch angle (ψ) in radians.
            yaw (float): Yaw angle (φ) in radians.
        
        Returns:
            numpy.ndarray: 3x3 rotation matrix.
        """
        
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        R = np.array([
            [cy*cp,         cy*sp*sr - sy*cr,    cy*sp*cr + sy*sr],
            [sy*cp,         sy*sp*sr + cy*cr,    sy*sp*cr - cy*sr],
            [  -sp,                  cp*sr,             cp*cr    ]
        ])

        return R
    
    def get_thrust_direction(self):
        """
        Compute the direction of the thrust vector in the earth-fixed frame.
        Since the thrust is along the body z-axis, we multiply the constant vector Θ3 = [0, 0, 1]^T
        by the transformation matrix T1 (which is the same as the rotation matrix R here).
        
        Returns:
            numpy.ndarray: 3x1 vector representing the direction of the thrust in the inertial frame.
        """

        roll, pitch, yaw = self.euler
        R = self.rotation_matrix(roll, pitch, yaw)
        thrust_direction = R[:, 2]
        return thrust_direction
    
    def compute_total_thrust(self, rotor_speeds):
        """
        Compute the total thrust P̄.
        
        According to the model: P̄ = p_const * sum(P_k^2)
        where rotor_speeds is an array containing the speeds (P_k) of each rotor.
        
        Parameters:
            rotor_speeds (array-like): Rotor speeds.
        
        Returns:
            float: Total thrust.
        """
        
        rotor_speeds = np.array(rotor_speeds)
        total_thrust = self.p_const * np.sum(rotor_speeds**2)
        return total_thrust

    def update_dynamics(self, rotor_speeds, dt, disturbance=np.zeros(3)):
        """
        Update the UAV dynamics over a time step dt.
        
        The dynamics (from Eq. (1)) are:
            1) dot(Δ) = Φ
            2) dot(Φ) = (P̄/M) * (T1 * Θ3) - G * Θ3 - (Γ/M) * Φ + disturbance
        
        Here, Θ3 = [0, 0, 1]^T and T1 * Θ3 is computed via the rotation matrix.
        An optional disturbance (e.g., lumped external disturbance) can be added.
        
        Parameters:
            rotor_speeds (array-like): Speeds of the six rotors.
            dt (float): Time step for integration.
            disturbance (numpy.ndarray): 3x1 vector of disturbances (default is zero).
        
        Returns:
            numpy.ndarray: Updated state vector [x, y, z, vx, vy, vz].
        """
        
        # Compute total thrust.
        P_bar = self.compute_total_thrust(rotor_speeds)
        
        # Get the thrust direction in the inertial frame.
        thrust_dir = self.get_thrust_direction()
        
        # Current velocity.
        velocity = self.state[3:]
        
        # Compute acceleration:
        acceleration = (P_bar / self.M) * thrust_dir - np.array([0, 0, self.G]) - (np.dot(self.Gamma, velocity) / self.M) + disturbance
        
        # Update position and velocity using Euler integration.
        self.state[:3] += velocity * dt        # Update position.
        self.state[3:] += acceleration * dt    # Update velocity.
        
        return self.state