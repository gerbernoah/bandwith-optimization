import json
import os
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD

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
=======
from typing import List


def plot_parameter_performance(param_name, file_names: List[str], nruns: List[int], optimal_cost: float):
    """
    Plot average error percentage from optimal cost for different parameter values

    Args:
    - param_name: Name of parameter to analyze (e.g., "cxpb")
    """
    # Construct file path

    list_of_values = []
    list_of_results = []
    for file_name in file_names:
        filepath = os.path.join("output", f"{file_name}.txt")

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No results file found for parameter: {file_name}")

        # Read results file
        with open(filepath, 'r') as f:
            lines = f.readlines()
            list_of_values.append(json.loads(lines[1].strip()))
            list_of_results.append(json.loads(lines[2].strip()))

    # Calculate average error percentages

    plt.figure(figsize=(10, 6))

    for i, (values, results, file_name) in enumerate(zip(list_of_values, list_of_results, file_names)):
        if len(values) != len(results):
            raise ValueError(
                f"Values and results length mismatch for parameter: {file_name[i]}")
        avg_costs = [np.mean(run_results) for run_results in results]
        error_percentages = [
            100 * ((cost - optimal_cost) / optimal_cost) for cost in avg_costs]

        # Plot all parameters on the same axes with different colors
        plt.plot(values, error_percentages, 'o-',
                 linewidth=2, markersize=8, label=nruns[i])

        # Add data labels
        for j, (val, err) in enumerate(zip(values, error_percentages)):
            plt.text(val, err, f"{err:.1f}%",
                     ha='center', va='bottom', fontsize=9)

    # Add labels and title (once for the entire plot)
    plt.xlabel('Parameter Value', fontsize=12)
    plt.xscale('log') if (param_name=="penalty_coeff") else None
    plt.ylabel('Average Error % from Optimal', fontsize=12)
    plt.title(param_name, fontsize=14)

    # Add grid and optimal reference line
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3, label='Optimal')

    plt.legend()
    plt.tight_layout()

    # Save and show
    plot_path = os.path.join("plots", f"{param_name}_parameters_performance.png")
    plt.savefig(plot_path, dpi=300)
    return plot_path


def plot_parameter_runtime(param_name: str, file_names: List[str], nruns: List[int], benchmark: float):
    """
    Plot average runtime for different parameter values

    Args:
    - param_names: List of parameter names to analyze (e.g., ["cxpb", "mutpb"])
    """
    # Construct file path
    list_of_values = []
    list_of_runtimes = []
    for file_name in file_names:
        filepath = os.path.join("output", f"{file_name}.txt")

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No results file found for parameter: {file_name}")

        # Read results file
        with open(filepath, 'r') as f:
            lines = f.readlines()
            list_of_values.append(json.loads(lines[1].strip()))
            list_of_runtimes.append(json.loads(lines[3].strip()))

    plt.figure(figsize=(10, 6))

    for i, (values, runtimes, file_name) in enumerate(zip(list_of_values, list_of_runtimes, file_names)):
        if len(values) != len(runtimes):
            raise ValueError(
                f"Values and runtimes length mismatch for parameter: {file_name}")

        # Calculate average runtimes
        avg_runtimes = [np.mean(run_times) for run_times in runtimes]

        # Plot all parameters on the same axes with different colors
        plt.plot(values, avg_runtimes, 's-', linewidth=2,
                 markersize=8, label=nruns[i])

        # Add data labels
        for j, (val, runtime) in enumerate(zip(values, avg_runtimes)):
            plt.text(val, runtime, f"{runtime:.1f}s",
                     ha='center', va='bottom', fontsize=9)

    # Add labels and title (once for the entire plot)
    plt.xlabel('Parameter Value', fontsize=12)
    plt.xscale('log') if (param_name=="penalty_coeff") else None
    plt.ylabel('Average Runtime (seconds)', fontsize=12)
    plt.title(param_name, fontsize=14)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=benchmark, color='r', linestyle='-', alpha=0.3, label='Integer Programming')

    plt.legend()
    plt.tight_layout()

    # Save and show
    plot_path = os.path.join("plots", f"{param_name}_runtime_analysis.png")
    plt.savefig(plot_path, dpi=300)
    return plot_path
>>>>>>> GA_optimization
