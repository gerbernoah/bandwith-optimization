from algorithms.integer_programming import run_IP
from utilities.parsing import parse_network
from utilities.printing import print_summary

network = parse_network()

results = run_IP(network)

print_summary("IP", results)

# from algorithms.genetic_algorithm import *

# main()