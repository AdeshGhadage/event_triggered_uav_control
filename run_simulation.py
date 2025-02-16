#!/usr/bin/env python3

from UAV import SixRotorUAV, FDICompensator, run_simulation, plot_results

def main():
    print("Starting UAV simulation...")
    
    # Run basic simulation
    print("Running basic simulation...")
    results = run_simulation(total_time=2.0, dt=0.01)
    
    # Plot the results
    print("Plotting results...")
    plot_results(results)
    
    # Example of accessing UAV and FDI components directly if needed
    print("\nDemonstrating direct component access:")
    uav = SixRotorUAV(mass=6.0, g=9.81, air_resistance=(0.1, 0.1, 0.1), p_const=0.05)
    fdi = FDICompensator(num_rotors=6)
    
    # Show component initialization
    print(f"UAV mass: {uav.M} kg")
    print(f"Number of rotors: {fdi.num_rotors}")
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()