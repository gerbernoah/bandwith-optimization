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
resultsIP = run_IP(network, timeLimit=3*60)
print_summary("IP", resultsIP)

"""
==================================================
    GENETIC ALGORITHM
==================================================
"""
# results = run_GA(network, ngen=50)
# print_summary("GA", results)
"""
base_params = GAParams(
    npop=80,
    cxpb=0.6,
    mutpb=0.2,
    ngen=40,
    penalty_coeff=100000.0,
    max_module_ratio=0.4,
    min_module_ratio=0.2,
    muLambda=(1, 1.7),
    indp=0.4,
    tournsize=3
)

# number of runs per param value
# nruns = [18, 19, 20]
nruns = [3, 4, 5]

# Define param values
cxpb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
mutpb = [0.1, 0.2, 0.3, 0.4]
# mutpb = [0, 0.02, 0.04, 0.06, 0.8]
penalty_coeff = [10, 100, 1000, 10000, 100000, 1000000]
tournsize = [2, 3, 4, 5]
indp = [0.1, 0.2, 0.3, 0.4]

# Run GA
param_lists = [cxpb, mutpb, penalty_coeff, tournsize, indp]
param_lists_names = ["cxpb", "mutpb", "penalty_coeff", "tournsize", "indp"]
for i in range(len(param_lists)):

    file_names = []
    for j in range(len(nruns)):
        file_name = param_lists_names[i] + "_pop_" + str(nruns[j])
        file_names.append(file_name)
        run_parameter_analysis(network, file_name, base_params, param_lists_names[i], param_lists[i], nruns[j])


    plot_parameter_performance(param_lists_names[i], file_names, nruns, resultsIP.total_cost)
    plot_parameter_runtime(param_lists_names[i], file_names, nruns, resultsIP.total_runtime)
"""

params = GAParams(
    npop=80,
    cxpb=0.8,
    mutpb=0.2,
    ngen=40,
    penalty_coeff=100000.0,
    max_module_ratio=0.4,
    min_module_ratio=0.2,
    muLambda=(1, 1.7),
    indp=0.4,
    tournsize=3
)

# number of runs per param value
# nruns = [18, 19, 20]
runs = 10

results = []
runtimes = []
print("==== running GA analysis ====")
for j in range(runs):
    result = run_GA(network, params, False)
    print(f"Run ({j+1}/{runs}): {result.total_cost}, {result.total_runtime}s")
    results.append(result.total_cost)
    runtimes.append(result.total_runtime)

error_percentages = [100 * ((cost - resultsIP.total_cost) / resultsIP.total_cost) for cost in results]
avgerror = sum(error_percentages) / len(error_percentages)
avgruntime = sum(runtimes) / len(runtimes)

print(f"\nAverage Results: avgerror = {avgerror}, avgruntime = {avgruntime}s")