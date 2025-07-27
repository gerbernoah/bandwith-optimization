import random
import numpy as np
from deap import base, creator, tools, algorithms
import copy

# Problem data (replace with actual data)
global_n_edges = 10
global_n_demands = 5
global_modules = [
    [(100, 50), (200, 100)],  # Edge 0
    [(150, 75)],               # Edge 1
    [(120, 60), (180, 90)],    # Edge 2
    [(80, 40)],                # Edge 3
    [(90, 45), (170, 85)],     # Edge 4
    [(110, 55)],               # Edge 5
    [(130, 65)],               # Edge 6
    [(140, 70), (210, 105)],   # Edge 7
    [(160, 80)],               # Edge 8
    [(70, 35), (190, 95)]      # Edge 9
]
global_demands = [
    (0, 1, 30),  # (source, target, value)
    (1, 2, 40),
    (2, 3, 20),
    (3, 4, 50),
    (4, 0, 25)
]
global_demand_paths = [
    [[0, 1], [0, 5, 2], [0, 6], [0, 7, 3], [0, 8, 4]],  # Demand 0
    [[1, 2], [1, 5, 3], [1, 6, 4], [1, 7], [1, 8, 9]],  # Demand 1
    [[2, 3], [2, 5, 4], [2, 6, 9], [2, 7, 8], [2, 1, 0]],  # Demand 2
    [[3, 4], [3, 5, 9], [3, 6, 0], [3, 7, 1], [3, 8, 2]],  # Demand 3
    [[4, 0], [4, 5, 1], [4, 6, 2], [4, 7, 3], [4, 8]]     # Demand 4
]
global_edge_routing_cost = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.9, 0.1, 0.5]

penalty_coeff = 1000.0  # Penalty for capacity violations

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def create_individual():
    # Module part: for each edge, choose a module or -1
    module_part = []
    for i in range(global_n_edges):
        choices = [-1] + list(range(len(global_modules[i])))
        module_part.append(random.choice(choices))
    
    # Flow part: for each demand, generate normalized fractions
    flow_part = []
    for _ in range(global_n_demands):
        # Generate random fractions using Dirichlet distribution
        fracs = np.random.dirichlet(np.ones(5), size=1)[0].tolist()
        flow_part.append(fracs)
    
    return creator.Individual([module_part, flow_part])

def cx_uniform(ind1, ind2):
    """Uniform crossover operator"""
    # Create deep copies to avoid reference issues
    mod1 = copy.deepcopy(ind1[0])
    mod2 = copy.deepcopy(ind2[0])
    flow1 = copy.deepcopy(ind1[1])
    flow2 = copy.deepcopy(ind2[1])
    
    # Module crossover: independent per edge
    for i in range(len(mod1)):
        if random.random() < 0.5:
            mod1[i], mod2[i] = mod2[i], mod1[i]
    
    # Flow crossover: independent per demand
    for j in range(len(flow1)):
        if random.random() < 0.5:
            flow1[j], flow2[j] = flow2[j], flow1[j]
    
    # Create new individuals
    new_ind1 = creator.Individual([mod1, flow1])
    new_ind2 = creator.Individual([mod2, flow2])
    
    return new_ind1, new_ind2

def mut_custom(individual, indpb):
    """Mutation operator with independent per-component mutation"""
    mod_part, flow_part = individual[0], individual[1]
    
    # Mutate module selections
    for i in range(len(mod_part)):
        if random.random() < indpb:
            choices = [-1] + list(range(len(global_modules[i])))
            mod_part[i] = random.choice(choices)
    
    # Mutate flow distributions
    for j in range(len(flow_part)):
        if random.random() < indpb:
            # Generate new normalized fractions
            new_fracs = np.random.dirichlet(np.ones(5), size=1)[0].tolist()
            flow_part[j] = new_fracs
    
    return individual,

def evaluate_individual(individual):
    """Fitness evaluation with capacity constraints"""
    module_sel, flow_fracs = individual[0], individual[1]
    total_cost = 0.0
    
    # 1. Calculate setup costs
    for i in range(global_n_edges):
        mod_idx = module_sel[i]
        if mod_idx != -1:
            total_cost += global_modules[i][mod_idx][0]  # Setup cost
    
    # 2. Initialize edge flows
    edge_flows = [0.0] * global_n_edges
    
    # 3. Calculate flows per demand
    for j in range(global_n_demands):
        demand_val = global_demands[j][2]
        for path_idx in range(5):
            flow_val = flow_fracs[j][path_idx] * demand_val
            # Add flow to all edges in this path
            for edge_idx in global_demand_paths[j][path_idx]:
                if edge_idx < global_n_edges:  # Ensure valid edge index
                    edge_flows[edge_idx] += flow_val
    
    # 4. Calculate routing costs and penalties
    for i in range(global_n_edges):
        # Get module capacity (0 if no module)
        capacity = 0
        if module_sel[i] != -1:
            capacity = global_modules[i][module_sel[i]][1]
        
        # Add routing cost
        total_cost += edge_flows[i] * global_edge_routing_cost[i]
        
        # Add penalty for capacity violation
        if edge_flows[i] > capacity:
            total_cost += penalty_coeff * (edge_flows[i] - capacity)
    
    return (total_cost,)  # Return as a tuple

# Register components with DEAP
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cx_uniform)
toolbox.register("mutate", mut_custom, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_individual)

def main():
    random.seed(42)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    
    # Fixed statistics - extract single fitness value
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    # Run GA with elitism
    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100,
        stats=stats, halloffame=hof, verbose=True
    )
    
    best_ind = hof[0]
    print("\nBest solution found:")
    print("Module selections:", best_ind[0])
    print("Total cost:", best_ind.fitness.values[0])
    
    # Calculate capacity utilization
    print("\nEdge utilization:")
    module_sel, flow_fracs = best_ind[0], best_ind[1]
    edge_flows = [0.0] * global_n_edges
    for j in range(global_n_demands):
        demand_val = global_demands[j][2]
        for path_idx in range(5):
            flow_val = flow_fracs[j][path_idx] * demand_val
            for edge_idx in global_demand_paths[j][path_idx]:
                if edge_idx < global_n_edges:
                    edge_flows[edge_idx] += flow_val
    
    for i in range(global_n_edges):
        capacity = 0
        if module_sel[i] != -1:
            capacity = global_modules[i][module_sel[i]][1]
        print(f"Edge {i}: Flow={edge_flows[i]:.2f}/{capacity} ", 
              f"(Violation: {max(0, edge_flows[i]-capacity):.2f})" if edge_flows[i] > capacity else "")
    
    return best_ind, log

if __name__ == "__main__":
    best_individual, logbook = main()