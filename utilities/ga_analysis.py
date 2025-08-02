import os
from typing import List, Union
import copy

from types2.network import Network
from types2.biology import GAParams, GAIndividual
from utilities.parsing import write_parameter_results
from algorithms.genetic_algorithm import run_GA


def run_parameter_analysis(
    network: Network,
    file_name: str,
    base_params: GAParams,
    param_name: str,
    param_values: List[Union[int, float]],
    n_runs: int = 5
):
    """
    Run GA multiple times with different values for a specific parameter. 
    Saves result in output folder as txt file.
    Args:
    - network: Network instance
    - base_params: Base parameters with default values
    - param_name: Name of parameter to analyze (e.g., "cxpb")
    - values: List of values to test for the parameter
    - n_runs: Number of runs per parameter value
    - results_dir: Directory to save results
    """

    # Prepare results storage
    all_results = []
    all_runtimes = []

    # Run experiments
    for value in param_values:
        print(f"\n=== Running {param_name} = {value} ===")
        value_results = []
        value_runtimes = []
        for run in range(n_runs):
            # Create parameter set with modified value
            params = copy.deepcopy(base_params)
            setattr(params, param_name, value)

            # Run GA
            result = run_GA(network, params, log=False)

            # Store results
            value_results.append(result.total_cost)
            value_runtimes.append(result.total_runtime)

            print(f"Run {run+1}/{n_runs}: Cost = {result.total_cost:.2e}, Violoations: {result.violations}, "
                  f"Time = {result.total_runtime:.2f}s")

        all_results.append(sum(value_results) / len(value_results))
        all_runtimes.append(sum(value_runtimes) / len(value_runtimes))

    # Save results
    write_parameter_results(
        file_name,
        base_params,
        param_values,
        all_results,
        all_runtimes
    )
