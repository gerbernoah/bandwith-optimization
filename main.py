from algorithms.integer_programming import run_IP
from algorithms.genetic_algorithm import run_GA
from utilities.parsing import parse_network
from utilities.printing import print_summary

"""
==================================================
    INPUT PARSING
==================================================
"""
network = parse_network()

"""
==================================================
    INTEGER PROGRAMMING
==================================================
"""
# results = run_IP(network, timeLimit=10)
# print_summary("IP", results)

"""
==================================================
    GENETIC ALGORITHM
==================================================
"""
results = run_GA(network, ngen=10)
print_summary("GA", results)