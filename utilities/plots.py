import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_parameter_performance(param_name: str, optimal_cost: float):
    """
    Plot average error percentage from optimal cost for different parameter values
    
    Args:
    - param_name: Name of parameter to analyze (e.g., "cxpb")
    - optimal_cost: Known optimal cost or best solution to compare against
    - results_dir: Directory containing results files
    """
    # Construct file path
    filepath = os.path.join("output", f"{param_name}.txt")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No results file found for parameter: {param_name}")
    
    # Read results file
    with open(filepath, 'r') as f:
        lines = f.readlines()
        values = json.loads(lines[1].strip())
        results = json.loads(lines[2].strip())
    
    # Calculate average error percentages
    avg_costs = [np.mean(run_results) for run_results in results]
    error_percentages = [100 * (cost - optimal_cost) / optimal_cost for cost in avg_costs]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(values, error_percentages, 'o-', linewidth=2, markersize=8)
    plt.xscale('log') if param_name=="penalty_coeff" else None # logarithmic for penalty coeff
    
    # Add labels and title
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Average Error % from Optimal', fontsize=12)
    plt.title(f"Parameter Analysis: {param_name}", fontsize=14)
    
    # Add grid and optimal reference line
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3, label='Optimal')
    
    # Add data labels
    for i, (val, err) in enumerate(zip(values, error_percentages)):
        plt.text(val, err, f"{err:.1f}%", 
                 ha='center', va='bottom', fontsize=9)
    
    plt.legend()
    plt.tight_layout()
    
    # Save and show
    plot_path = os.path.join("plots", f"{param_name}_performance.png")
    plt.savefig(plot_path, dpi=300)
    return plot_path

def plot_parameter_runtime(param_name: str):
    """
    Plot average runtime for different parameter values
    
    Args:
    - param_name: Name of parameter to analyze (e.g., "cxpb")
    - results_dir: Directory containing results files
    """
    # Construct file path
    filepath = os.path.join("output", f"{param_name}.txt")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No results file found for parameter: {param_name}")
    
    # Read results file
    with open(filepath, 'r') as f:
        lines = f.readlines()
        values = json.loads(lines[1].strip())
        runtimes = json.loads(lines[3].strip())  # Runtime data is on line 4
    
    # Calculate average runtimes
    avg_runtimes = [np.mean(run_times) for run_times in runtimes]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(values, avg_runtimes, 's-', color='green', linewidth=2, markersize=8)
    plt.xscale('log') if param_name=="penalty_coeff" else None # logarithmic for penalty coeff
    
    # Add labels and title
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Average Runtime (seconds)', fontsize=12)
    plt.title(f"Runtime Analysis: {param_name}", fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, (val, runtime) in enumerate(zip(values, avg_runtimes)):
        plt.text(val, runtime, f"{runtime:.1f}s", 
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save and show
    plot_path = os.path.join("plots", f"{param_name}_runtime.png")
    plt.savefig(plot_path, dpi=300)
    return plot_path