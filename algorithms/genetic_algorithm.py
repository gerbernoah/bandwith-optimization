import time
import numpy as np
import pygad

# this import works when this code is run from root directory
from types.network import NetworkGraph
from types.result import ResultGA

class GeneticAlgorithmOptimizerSimple:
    """Minimal Genetic Algorithm for network optimization (no history, only cost/runtime)"""

    def __init__(self, network: NetworkGraph, time_limit=None, msg=False):
        self.network = network
        self.link_ids = list(network.links.keys())
        self.num_genes = len(self.link_ids)
        self.time_limit = time_limit
        self.start_time = None
        self.timeout_occurred = False
        self.msg = msg

    def create_initial_population(self, sol_per_pop: int) -> np.ndarray:
        population = []
        for _ in range(sol_per_pop):
            solution = []
            for link_id in self.link_ids:
                link = self.network.links[link_id]
                num_options = len(link.get_total_capacity_options())
                choice = np.random.randint(0, num_options)
                solution.append(choice)
            population.append(solution)
        return np.array(population)

    def fitness_function(self, ga_instance, solution, solution_idx):
        total_cost = 0.0
        penalty = 0.0
        for i, link_id in enumerate(self.link_ids):
            link = self.network.links[link_id]
            module_choice = int(solution[i])
            capacity_options = link.get_total_capacity_options()
            if module_choice < len(capacity_options):
                capacity, cost = capacity_options[module_choice]
                total_cost += cost
                demand = self._get_link_demand(link_id)
                if demand > capacity:
                    penalty += (demand - capacity) * 1e9
            else:
                penalty += 1e10
        baseline = 1e8
        fitness = baseline - total_cost - penalty
        return fitness

    def _get_link_demand(self, link_id: str) -> float:
        link = self.network.links[link_id]
        demand = 0.0
        for demand_obj in self.network.demands.values():
            if (demand_obj.source == link.source and demand_obj.target == link.target) or \
               (demand_obj.source == link.target and demand_obj.target == link.source):
                demand += demand_obj.demand_value
        return demand

    def optimize(self, num_generations=100, sol_per_pop=50, num_parents_mating=25):
        self.start_time = time.time()
        self.timeout_occurred = False
        initial_population = self.create_initial_population(sol_per_pop)
        gene_space = []
        for link_id in self.link_ids:
            link = self.network.links[link_id]
            num_options = len(link.get_total_capacity_options())
            gene_space.append(list(range(num_options)))
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=self.fitness_function,
            sol_per_pop=sol_per_pop,
            num_genes=self.num_genes,
            initial_population=initial_population,
            gene_space=gene_space,
            parent_selection_type="sss",
            keep_parents=1,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10
        )
        try:
            ga_instance.run()
        except (KeyboardInterrupt, TimeoutError):
            self.timeout_occurred = True
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        total_time = time.time() - self.start_time if self.start_time else 0
        actual_cost = self.get_solution_cost(solution)
        return ResultGA(
            total_runtime=total_time,
            total_cost=actual_cost
        )

    def get_solution_cost(self, solution):
        total_cost = 0.0
        for i, link_id in enumerate(self.link_ids):
            link = self.network.links[link_id]
            module_choice = int(solution[i])
            capacity_options = link.get_total_capacity_options()
            if module_choice < len(capacity_options):
                _, cost = capacity_options[module_choice]
                total_cost += cost
        return total_cost

def run_genetic_algorithm_simple(nodes_file="nodes.txt", links_file="links.txt", demands_file="demands.txt",
                                 num_generations=50, sol_per_pop=50, num_parents_mating=20, time_limit=None, msg=False):
    network = NetworkGraph()
    network.load_from_files(nodes_file, links_file, demands_file)
    optimizer = GeneticAlgorithmOptimizerSimple(network, time_limit=time_limit, msg=msg)
    return optimizer.optimize(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating
    )