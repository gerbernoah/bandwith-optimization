"""
Functions for printing optimization results.

functions: print_summary
"""
# this works when this code is run from root directory
from types2.algorithms import algorithm_type
from types2.result import ResultGA, ResultIP

algorithm_name = {"GA": "Genetic Algorithm", "IP": "Integer Programming"}

def print_title(title: str):
    l = f"\n{'='*60}"
    print(f"{l}\n{title}{l}")

def print_summary(algorithm: algorithm_type, results: ResultGA | ResultIP):
    """Print optimization summary for ResultGA or ResultIP objects."""

    print_title(f"{algorithm_name[algorithm]} OPTIMIZATION RESULTS")

    if algorithm == "GA":
        print(f"Total Runtime: {results.total_runtime:.2f} seconds")
        print(f"Total Cost: ${results.total_cost:,.2f}")

    elif algorithm == "IP":
        print(f"Total Runtime: {results.total_runtime:.2f} seconds")
        print(f"Total Cost: ${results.total_cost:,.2f}")

    else:
        print("[Warning] Unknown result type. No details to show.")