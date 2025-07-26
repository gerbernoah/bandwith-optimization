#!/usr/bin/env python3
"""
Minimized Optimization Caller

This module runs both genetic algorithm and integer programming optimization
and creates shared comparison plots.

Usage:
    python main.py [genetic|ip|both]
"""

# Import: libraries
import concurrent.futures

# Import: types, utilities, algorithms
from types.result import ResultGA, ResultIP
from types.algorithms import algorithm_choice
from utilities.printing import print_summary
from algorithms.genetic_algorithm import run_genetic_algorithm
from algorithms.integer_programming import run_integer_programming

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

"""
===================================================
 Main Function
===================================================
"""

def main(algorithm: algorithm_choice = "both"):
    if algorithm == "GA":
        print("Running Genetic Algorithm...")
        results = genetic_algorithm()
        print_summary("GA", results)

    elif algorithm == "IP":
        print("Running Integer Programming...")
        results = integer_programming()
        print_summary("IP", results)

    else:
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
            ga_results: ResultGA = ga_future.result()
            ip_results: ResultIP = ip_future.result()

        print_summary("GA", ga_results)
        print_summary("IP", ip_results)

if __name__ == "__main__":
    main("both")
