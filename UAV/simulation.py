import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
from .six_rotor_uav import SixRotorUAV
from .disturbance_observer import DisturbanceObserver
from .fdi_compensator import FDICompensator

def run_simulation(total_time=2.0, dt=0.01):
    """
    Run a complete UAV simulation with disturbance and fault injection.
    
    Parameters:
        total_time (float): Total simulation time in seconds
        dt (float): Time step in seconds
    
    Returns:
        dict: Simulation results containing states, times, and other data
    """
    time_steps = int(total_time / dt)
    times = np.linspace(0, total_time, time_steps)
    
    # Initialize UAV and components
    uav = SixRotorUAV(mass=6.0, g=9.81, air_resistance=(0.1, 0.1, 0.1), p_const=0.05)
    observer = DisturbanceObserver(mass=6.0)
    fdi = FDICompensator(num_rotors=6)
    
    # Set initial conditions
    uav.set_euler_angles(roll=0.1, pitch=0.05, yaw=0.2)
    base_rotor_speeds = np.array([400, 400, 400, 400, 400, 400])
    
    # Initialize storage arrays
    states = []
    disturbances = []
    rotor_states = []
    
    # Add artificial disturbance at t=1.0s
    disturbance_time = 1.0
    
    # Simulate
    for t in times:
        # Add artificial wind disturbance
        if t >= disturbance_time:
            disturbance = np.array([0.5, 0.3, 0.0])  # Wind force
        else:
            disturbance = np.zeros(3)
            
        # Update UAV dynamics
        state = uav.update_dynamics(base_rotor_speeds, dt, disturbance)
        
        # Estimate disturbance
        thrust_dir = uav.get_thrust_direction()
        total_thrust = uav.compute_total_thrust(base_rotor_speeds)
        est_disturbance = observer.update(state, thrust_dir, total_thrust, dt)
        
        # Store results
        states.append(state.copy())
        disturbances.append(est_disturbance.copy())
        
    return {
        'times': times,
        'states': np.array(states),
        'disturbances': np.array(disturbances)
    }

def plot_results(results):
    """
    Create visualization plots for the simulation results.
    
    Parameters:
        results (dict): Simulation results from run_simulation()
    """
    times = results['times']
    states = results['states']
    disturbances = results['disturbances']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(states[:, 0], states[:, 1], states[:, 2])
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('UAV 3D Trajectory')
    
    # Position plots
    ax2 = fig.add_subplot(222)
    ax2.plot(times, states[:, 0], label='X')
    ax2.plot(times, states[:, 1], label='Y')
    ax2.plot(times, states[:, 2], label='Z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    
    # Velocity plots
    ax3 = fig.add_subplot(223)
    ax3.plot(times, states[:, 3], label='VX')
    ax3.plot(times, states[:, 4], label='VY')
    ax3.plot(times, states[:, 5], label='VZ')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs Time')
    ax3.legend()
    
    # Disturbance plots
    ax4 = fig.add_subplot(224)
    ax4.plot(times, disturbances[:, 0], label='X')
    ax4.plot(times, disturbances[:, 1], label='Y')
    ax4.plot(times, disturbances[:, 2], label='Z')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Disturbance Force (N)')
    ax4.set_title('Estimated Disturbances')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Run simulation
    results = run_simulation(total_time=2.0, dt=0.01)
    
    # Plot results
    plot_results(results)