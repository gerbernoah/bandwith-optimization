import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_parameter_performance(param_names: List[str], optimal_cost: float):
    """
    Plot average error percentage from optimal cost for different parameter values

    Args:
    - param_name: Name of parameter to analyze (e.g., "cxpb")
    - optimal_cost: Known optimal cost or best solution to compare against
    - results_dir: Directory containing results files
    """
    # Construct file path

    list_of_values = []
    list_of_results = []
    for param_name in param_names:
        filepath = os.path.join("output", f"{param_name}.txt")

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No results file found for parameter: {param_name}")

        # Read results file
        with open(filepath, 'r') as f:
            lines = f.readlines()
            list_of_values.append(json.loads(lines[1].strip()))
            list_of_results.append(json.loads(lines[2].strip()))

    # Calculate average error percentages

    plt.figure(figsize=(10, 6))

    for i, (values, results, param_name) in enumerate(zip(list_of_values, list_of_results, param_names)):
        if len(values) != len(results):
            raise ValueError(
                f"Values and results length mismatch for parameter: {param_names[i]}")
        avg_costs = [np.mean(run_results) for run_results in results]
        error_percentages = [
            100 * (cost - optimal_cost) / optimal_cost for cost in avg_costs]

        # Plot all parameters on the same axes with different colors
        plt.plot(values, error_percentages, 'o-',
                 linewidth=2, markersize=8, label=param_name)

        # Add data labels
        for j, (val, err) in enumerate(zip(values, error_percentages)):
            plt.text(val, err, f"{err:.1f}%",
                     ha='center', va='bottom', fontsize=9)

    # Add labels and title (once for the entire plot)
    plt.xlabel('Parameter Value', fontsize=12)
    plt.ylabel('Average Error % from Optimal', fontsize=12)
    plt.title('Parameter Analysis: All Parameters', fontsize=14)

    # Add grid and optimal reference line
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3, label='Optimal')

    plt.legend()
    plt.tight_layout()

    # Save and show
    name = param_names[0].split("_")[0] if param_names else "unknown"

    plot_path = os.path.join("plots", f"{name}_parameters_performance.png")
    plt.savefig(plot_path, dpi=300)
    return plot_path


def plot_parameter_runtime(param_names: List[str]):
    """
    Plot average runtime for different parameter values

    Args:
    - param_names: List of parameter names to analyze (e.g., ["cxpb", "mutpb"])
    """
    # Construct file path
    list_of_values = []
    list_of_runtimes = []
    for param_name in param_names:
        filepath = os.path.join("output", f"{param_name}.txt")

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No results file found for parameter: {param_name}")

        # Read results file
        with open(filepath, 'r') as f:
            lines = f.readlines()
            list_of_values.append(json.loads(lines[1].strip()))
            list_of_runtimes.append(json.loads(lines[3].strip()))

    plt.figure(figsize=(10, 6))

    for i, (values, runtimes, param_name) in enumerate(zip(list_of_values, list_of_runtimes, param_names)):
        if len(values) != len(runtimes):
            raise ValueError(
                f"Values and runtimes length mismatch for parameter: {param_name}")

        # Calculate average runtimes
        avg_runtimes = [np.mean(run_times) for run_times in runtimes]

        # Plot all parameters on the same axes with different colors
        plt.plot(values, avg_runtimes, 's-', linewidth=2,
                 markersize=8, label=param_name)

        # Add data labels
        for j, (val, runtime) in enumerate(zip(values, avg_runtimes)):
            plt.text(val, runtime, f"{runtime:.1f}s",
                     ha='center', va='bottom', fontsize=9)

    # Add labels and title (once for the entire plot)
    plt.xlabel('Parameter Value', fontsize=12)
    plt.ylabel('Average Runtime (seconds)', fontsize=12)
    plt.title('Runtime Analysis: All Parameters', fontsize=14)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend()
    plt.tight_layout()

    # Save and show
    name = param_names[0].split("_")[0] if param_names else "unknown"

    plot_path = os.path.join("plots", f"{name}_runtime_analysis.png")
    plt.savefig(plot_path, dpi=300)
    return plot_path
