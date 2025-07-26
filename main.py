#!/usr/bin/env python3
"""
Minimized Optimization Caller

This module runs both genetic algorithm and integer programming optimization
and creates shared comparison plots.

Usage:
    python main.py [genetic|ip|both]
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures

from result import ResultGA, ResultIP

# Clean imports using wrapper modules
from genetic.genetic_algorithm import run_genetic_algorithm
from integer.integer_programming import run_integer_programming


type algorithm_type = "GA" | "IP"
type algorithm_choice = "GA" | "IP" | "both"
algorithm_name = {"GA": "Genetic Algorithm", "IP": "Integer Programming"}


def print_summary(algorithm: algorithm_type, results: ResultGA | ResultIP):
    """Print optimization summary for ResultGA or ResultIP objects."""

    print(f"\n{'='*60}")
    print(f"{algorithm_name[algorithm]} OPTIMIZATION RESULTS")
    print(f"{'='*60}")

    if algorithm == "GA":
        print(f"Total Runtime: {results.total_runtime:.2f} seconds")
        print(f"Total Cost: ${results.total_cost:,.2f}")

    elif algorithm == "IP":
        print(f"Total Runtime: {results.total_runtime:.2f} seconds")
        print(f"Total Cost: ${results.total_cost:,.2f}")

    else:
        print("[Warning] Unknown result type. No details to show.")


def genetic_algorithm() -> ResultGA:
    return run_genetic_algorithm(
        num_generations=50,
        sol_per_pop=50,
        num_parents_mating=20,
        time_limit=1,
        msg=False
    )


def integer_programming() -> ResultIP:
    return run_integer_programming()


def main(algorithm: algorithm_choice = "both"):
    """Main function to run optimization algorit hms."""
    
    if algorithm == "GA":
        print("Running Genetic Algorithm...")
        results = genetic_algorithm()
        print_summary("GA", results)

    elif algorithm == "IP":
        print("Running Integer Programming...")
        results = integer_programming()
        print_summary("IP", results)

    else:  # both
        print("Running Both Algorithms in Parallel...")

        # Use ThreadPoolExecutor to run both algorithms concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            print("\n🧬 Starting Genetic Algorithm...")
            print("🔢 Starting Integer Programming...")

            # Submit both tasks
            ga_future = executor.submit(
                genetic_algorithm,
            )
            ip_future = executor.submit(
                integer_programming
            )

            # Wait for results
            print("\n⏳ Waiting for algorithms to complete...")
            ga_results = ga_future.result()
            ip_results = ip_future.result()

        print_summary("GA", ga_results)
        print_summary("IP", ip_results)

if __name__ == "__main__":
    main("both")  # Default to both algorithms
