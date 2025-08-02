from algorithms.integer_programming import run_IP
from algorithms.genetic_algorithm import run_GA
from utilities.parsing import parse_network
from utilities.printing import print_summary
from utilities.ga_analysis import run_parameter_analysis
from utilities.plots import plot_parameter_performance, plot_parameter_runtime
from types2.biology import GAParams

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
# results = run_GA(network, ngen=50)
# print_summary("GA", results)

base_params = GAParams(
    npop=10,
    cxpb=0.8,
    mutpb=0.2,
    ngen=10,
    penalty_coeff=100000.0,
    max_module_ratio=0.4,
    min_module_ratio=0.2,
    muLambda=(1, 1.7),
    indp=0.4,
    tournsize=3
)

# population
pops = [18, 19, 20]

# Define param values
cxpb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
mutpb = [0.1, 0.2]
penalty_coeff = [10, 100, 1000, 10000, 100000, 1000000]

# Run GA
optimum = 30000000
param_lists = [mutpb, penalty_coeff]
param_lists_names = ["mutpb", "penalty_coeff"]
for i in range(len(param_lists)):

    file_names = []
    for j in range(len(pops)):
        file_name = param_lists_names[i] + "_pop_" + str(pops[j])
        file_names.append(file_name)
        run_parameter_analysis(
            network, file_name, base_params, param_lists_names[i], param_lists[i], 3)

    plot_parameter_performance(file_names, optimum)
    plot_parameter_runtime(file_names)

plot_parameter_runtime(["penalty_coeff"])
plot_parameter_performance(["penalty_coeff"], optimum)
