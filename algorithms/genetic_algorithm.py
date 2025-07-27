import random
import numpy as np
from deap import base, creator, tools, algorithms
import copy
import networkx as nx
from itertools import islice
from typing import List, Tuple
import time

# Data Structures & Utilities
from types2.network import Node, Demand, Edge, Module, NodeDict, UEdge, UEdgeToEdge, Network
from types2.biology import GAIndividual, Path, Paths5, DemandPaths
from types2.result import ResultGA
from utilities.printing import print_title

# Global variables for GA
global_demand_paths: DemandPaths = []  # Will store precomputed paths

"""
==================================================
    PRECOMPUTATION (demand paths)
==================================================
"""

def precompute_demand_paths(network: Network) -> DemandPaths:
    """Precompute 5 shortest paths for each demand based on routing cost"""
    _, _, edges, uedges, _, demands = network.unpack()
    
    # Build directed graph
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge.source.id, edge.target.id, 
                   weight=edge.uEdge.routing_cost, 
                   edge_id=edge.id)
    
    demand_paths: DemandPaths = []
    for demand in demands:
        source_id = demand.source.id
        target_id = demand.target.id
        
        try:
            # Compute up to 5 shortest paths
            paths = list(islice(nx.shortest_simple_paths(G, source_id, target_id, weight='weight'), 5))
        except (nx.NetworkXNoPath, nx.NetworkXError):
            paths = []
        
        # Convert node paths to edge ID paths
        edge_paths: List[Path] = []
        for path in paths:
            edge_ids = []
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                edge_id = G[u][v]['edge_id']
                edge_ids.append(edge_id)
            edge_paths.append([edges[id] for id in edge_ids])
        
        # Pad with empty paths if fewer than 5
        while len(edge_paths) < 5:
            edge_paths.append([])
        
        demand_paths.append(edge_paths)
    
    return demand_paths

"""
==================================================
    MODEL FUNCTIONS
==================================================
"""

def create_individual(uedges, demands) -> GAIndividual:
    """Create a GA individual with fitness attribute"""
    # Module part: for each undirected edge
    module_selections = []
    for uedge in uedges:
        choices = [-1] + list(range(len(uedge.module_options)))
        module_selections.append(random.choice(choices))
    
    # Flow part: for each demand
    flow_fractions = []
    for _ in range(len(demands)):
        fracs = np.random.dirichlet(np.ones(5), size=1)[0].tolist()
        flow_fractions.append(fracs)
    
    return GAIndividual(
        module_selections=module_selections,
        flow_fractions=flow_fractions
    )

def cx_uniform(ind1: GAIndividual, ind2: GAIndividual) -> Tuple[GAIndividual, GAIndividual]:
    """Uniform crossover operator"""
    # Module crossover
    mod1 = copy.deepcopy(ind1.module_selections)
    mod2 = copy.deepcopy(ind2.module_selections)
    for i in range(len(mod1)):
        if random.random() < 0.5:
            mod1[i], mod2[i] = mod2[i], mod1[i]
    
    # Flow crossover
    flow1 = copy.deepcopy(ind1.flow_fractions)
    flow2 = copy.deepcopy(ind2.flow_fractions)
    for j in range(len(flow1)):
        if random.random() < 0.5:
            flow1[j], flow2[j] = flow2[j], flow1[j]
    
    return (
        GAIndividual(module_selections=mod1, flow_fractions=flow1),
        GAIndividual(module_selections=mod2, flow_fractions=flow2)
    )

def mut_custom(individual: GAIndividual, indpb: float, uedges) -> Tuple[GAIndividual]:
    """Mutation operator that returns a tuple for DEAP compatibility"""
    # Mutate module selections
    for i in range(len(individual.module_selections)):
        if random.random() < indpb:
            choices = [-1] + list(range(len(uedges[i].module_options)))
            individual.module_selections[i] = random.choice(choices)
    
    # Mutate flow distributions
    for j in range(len(individual.flow_fractions)):
        if random.random() < indpb:
            new_fracs = np.random.dirichlet(np.ones(5), size=1)[0].tolist()
            individual.flow_fractions[j] = new_fracs
    
    return (individual,)  # Return as tuple

def evaluate_individual(individual: GAIndividual, network: Network, penalty_coeff: float) -> Tuple[float]:
    """Evaluate fitness using network data"""
    # Unpack network components
    nodes, node_dict, edges, uedges, uedge_to_edge, demands = network.unpack()
    total_cost = 0.0
    
    # 1. Calculate setup costs
    for i, uedge in enumerate(uedges):
        mod_idx = individual.module_selections[i]
        if mod_idx != -1:
            total_cost += uedge.module_options[mod_idx].cost
    
    # 2. Initialize edge flows
    directed_edge_flows = [0.0] * len(edges)
    
    # 3. Calculate flows per demand
    for j, demand in enumerate(demands):
        for path_idx in range(5):
            flow_val = individual.flow_fractions[j][path_idx] * demand.value
            for edge in global_demand_paths[j][path_idx]:
                directed_edge_flows[edge.id] += flow_val
    
    # 4. Aggregate flows to undirected edges
    uedge_flows = [0.0] * len(uedges)
    for edge in edges:
        uedge_flows[edge.uEdge.id] += directed_edge_flows[edge.id]
    
    # 5. Calculate routing costs and penalties
    routing_cost = 0.0
    for edge in edges:
        routing_cost += directed_edge_flows[edge.id] * edge.uEdge.routing_cost
    total_cost += routing_cost
    
    # 6. Check capacity constraints
    for i, uedge in enumerate(uedges):
        mod_idx = individual.module_selections[i]
        capacity = 0
        if mod_idx != -1:
            capacity = uedge.module_options[mod_idx].capacity
        
        if uedge_flows[i] > capacity:
            total_cost += penalty_coeff * (uedge_flows[i] - capacity)
    
    return (total_cost,)

def setup_toolbox(network: Network, penalty_coeff):
    """Setup DEAP toolbox with network-specific parameters"""
    _, _, _, uedges, _, demands = network.unpack()
    
    # DEAP setup with typed individual
    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: create_individual(uedges, demands))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
    toolbox.register("mate", cx_uniform)
    toolbox.register("mutate", mut_custom, indpb=0.1, uedges=uedges)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual, network=network, penalty_coeff=penalty_coeff)
    
    return toolbox

"""
==================================================
    RUN GENETIC ALGORITHM
==================================================
"""

def run_GA(
        network: Network,
        npop = 200,
        cxpb = 0.7,
        mutpb = 0.4,
        ngen = 200,
        penalty_coeff = 1000.0,
        log = True,
    ):
    """
    Args:
    - network: input data parsed as Network
    - npop: initial population size
    - cxpb: cross over probability
    - mutpb: mutation probability
    - ngen: number of generations
    - log: whether stats should be printed
    """
    print_title("GENETIC ALGORITHM")
    print("=== LOGGING ON ===" if log else "=== LOGGING OFF ===")

    """
    ==================================================
        SETUP & RUN
    ==================================================
    """

    # Parse network and precompute paths
    global global_demand_paths
    global_demand_paths = precompute_demand_paths(network)
    
    # Setup GA
    toolbox = setup_toolbox(network, penalty_coeff)
    _, _, _, uedges, _, demands = network.unpack()
    
    # Run GA
    random.seed(42)
    pop = toolbox.population(n=npop) # type: ignore
    hof = tools.HallOfFame(1)
    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    startTime = time.perf_counter()
    
    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
        stats=stats, halloffame=hof, verbose=log
    )

    endTime = time.perf_counter()
    runtime = endTime - startTime

    """
    ==================================================
        DISPLAY FLOW & VIOLATIONS
    ==================================================
    """
    
    # Display results
    best_ind = hof[0]
    print("\nBest solution found:")
    print("Total cost:", best_ind.fitness.values[0])
    
    # Detailed capacity utilization report
    _, _, edges, uedges, _, _ = network.unpack()
    module_sel, flow_fracs = best_ind.module_selections, best_ind.flow_fractions
    
    # Recalculate flows
    directed_edge_flows = [0.0] * len(edges)
    for j, demand in enumerate(demands):
        for path_idx in range(5):
            flow_val = flow_fracs[j][path_idx] * demand.value
            for edge in global_demand_paths[j][path_idx]:
                directed_edge_flows[edge.id] += flow_val
    
    uedge_flows = [0.0] * len(uedges)
    for edge in edges:
        uedge_id = edge.uEdge.id
        if uedge_id < len(uedge_flows):
            uedge_flows[uedge_id] += directed_edge_flows[edge.id]
    
    print("\nEdge utilization:")
    for i, uedge in enumerate(uedges):
        mod_idx = module_sel[i]
        capacity = 0
        if mod_idx != -1:
            capacity = uedge.module_options[mod_idx].capacity
        
        flow = uedge_flows[i]
        violation = max(0, flow - capacity)
        print(f"UEdge {i}: Flow={flow:.2f}/{capacity} ", 
              f"(Violation: {violation:.2f})" if violation > 0 else "")
    
    return ResultGA(
        total_runtime = runtime,
        total_cost = best_ind.fitness.values[0]
    )