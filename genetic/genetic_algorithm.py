import importlib
import os
import sys
import time
from typing import Dict
import numpy as np
import pygad


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
network_module = importlib.import_module('network')
NetworkGraph = network_module.NetworkGraph


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm implementation for network optimization"""

    def __init__(self, network: NetworkGraph, time_limit=None, msg=False):
        self.network = network
        self.link_ids = list(network.links.keys())
        self.num_genes = len(self.link_ids)
        self.time_limit = time_limit
        self.start_time = None
        self.timeout_occurred = False
        self.msg = msg

        # Calculate maximum module options per link
        if self.link_ids:
            self.max_options_per_link = max(
                len(link.get_total_capacity_options())
                for link in network.links.values()
            )
        else:
            self.max_options_per_link = 1

        self.best_solutions = []
        self.best_fitness_values = []
        self.generation_times = []
        self.average_fitness_values = []
        self.population_diversity = []
        self.convergence_data = []

    def create_initial_population(self, sol_per_pop: int) -> np.ndarray:
        """Create initial population where each gene represents ONE module choice per link.

        Gene values:
        - 0: No additional module (only pre-installed capacity)
        - 1: First module option
        - 2: Second module option
        - etc.

        This ensures only one module is selected per edge.
        """
        population = []

        for _ in range(sol_per_pop):
            solution = []
            for link_id in self.link_ids:
                link = self.network.links[link_id]
                num_options = len(link.get_total_capacity_options())
                # Random choice of module option (0 = no module, 1+ = specific module index)
                choice = np.random.randint(0, num_options)
                solution.append(choice)
            population.append(solution)

        return np.array(population)

    def fitness_function(self, ga_instance, solution, solution_idx):
        """Fitness function to minimize total cost while meeting capacity constraints"""
        total_cost = 0.0
        total_capacity_utilization = 0.0
        penalty = 0.0

        # Calculate cost and capacity for each link
        link_capacities = {}

        for i, link_id in enumerate(self.link_ids):
            link = self.network.links[link_id]
            module_choice = int(solution[i])

            capacity_options = link.get_total_capacity_options()

            if module_choice < len(capacity_options):
                capacity, cost = capacity_options[module_choice]
                total_cost += cost
                link_capacities[link_id] = capacity

                # Add routing cost based on capacity utilization
                demand = self._get_link_demand(link_id)
                if capacity > 0 and demand > 0:
                    utilization = min(1.0, demand / capacity)
                    total_cost += link.routing_cost * demand  # Cost per unit of demand
                    total_capacity_utilization += utilization
                elif demand > capacity:
                    # Capacity constraint violated - heavy penalty
                    penalty += (demand - capacity) * 10000
            else:
                # Invalid choice - add penalty
                penalty += 100000

        # Check for unmet demands (links with 0 capacity but positive demand)
        for link_id, demand in self._get_all_link_demands().items():
            if link_id in link_capacities and demand > 0:
                capacity = link_capacities[link_id]
                if capacity == 0:
                    # No capacity installed but demand exists
                    penalty += demand * 5000
                elif demand > capacity:
                    # Insufficient capacity
                    penalty += (demand - capacity) * 1000

        # Fitness should be maximized, so we need to transform the cost minimization problem
        # Use a large baseline value minus costs and penalties
        baseline = 100000000  # Large baseline value

        # Add bonus for efficient capacity utilization
        efficiency_bonus = 0
        if link_capacities:
            num_active_links = len(
                [c for c in link_capacities.values() if c > 0])
            if num_active_links > 0:
                avg_utilization = total_capacity_utilization / num_active_links
                efficiency_bonus = avg_utilization * 10000  # Bonus for good utilization

        # Higher fitness = better solution
        fitness = baseline - total_cost - penalty + efficiency_bonus

        return fitness

    def _get_link_demand(self, link_id: str) -> float:
        """Get traffic demand for a specific link using demand data"""
        link = self.network.links[link_id]
        # Sum demands that could use this link directly
        demand = 0.0
        for demand_obj in self.network.demands.values():
            # Direct connection
            if (demand_obj.source == link.source and demand_obj.target == link.target) or \
               (demand_obj.source == link.target and demand_obj.target == link.source):
                demand += demand_obj.demand_value
        return demand

    def _get_all_link_demands(self) -> Dict[str, float]:
        """Get traffic demands for all links"""
        demands = {}
        for link_id in self.link_ids:
            demands[link_id] = self._get_link_demand(link_id)
        return demands

    def on_generation(self, ga_instance):
        """Callback function called after each generation"""
        generation_start_time = time.time()

        # Check for timeout
        if self.time_limit and self.start_time:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.time_limit:
                # Check if we should ensure at least one iteration
                if ga_instance.generations_completed == 0:
                    if self.msg:
                        print(
                            f"⏰ Time limit reached ({elapsed_time:.2f}s / {self.time_limit}s)")
                        print("   Ensuring at least one generation completes...")
                    # Allow this generation to complete
                    pass
                else:
                    self.timeout_occurred = True
                    if self.msg:
                        print(
                            f"⏰ Time limit reached ({elapsed_time:.2f}s / {self.time_limit}s)")
                        print(
                            f"   Stopping after {ga_instance.generations_completed} generations")
                    # Raise an exception to stop the GA execution
                    raise TimeoutError("Time limit exceeded")

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        self.best_solutions.append(solution.copy())
        self.best_fitness_values.append(solution_fitness)

        # Print progress messages if msg is enabled
        if self.msg:
            actual_cost, penalty_cost = self.get_solution_cost(solution)
            print(f"Generation {ga_instance.generations_completed:3d}: "
                  f"Best Fitness = {solution_fitness:10.2f}, "
                  f"Actual Cost = ${actual_cost:8.2f}, "
                  f"Penalty = ${penalty_cost:8.2f}")

        # Track generation timing
        if self.start_time:
            generation_time = time.time() - self.start_time
            self.generation_times.append(generation_time)

        # Track average fitness of current population
        current_population_fitness = []
        for i in range(ga_instance.sol_per_pop):
            try:
                fitness = ga_instance.last_generation_fitness[i]
                current_population_fitness.append(fitness)
            except (IndexError, AttributeError):
                pass

        if current_population_fitness:
            avg_fitness = np.mean(current_population_fitness)
            self.average_fitness_values.append(avg_fitness)

            # Calculate population diversity (standard deviation of fitness)
            diversity = np.std(current_population_fitness)
            self.population_diversity.append(diversity)
        else:
            self.average_fitness_values.append(solution_fitness)
            self.population_diversity.append(0.0)

        # Track convergence data (improvement rate)
        if len(self.best_fitness_values) > 1:
            improvement = self.best_fitness_values[-1] - \
                self.best_fitness_values[-2]
            self.convergence_data.append(improvement)
        else:
            self.convergence_data.append(0.0)

    def optimize(self, num_generations=100, sol_per_pop=50, num_parents_mating=25):
        """Run the genetic algorithm optimization"""
        self.start_time = time.time()
        self.timeout_occurred = False

        # Print initial message if msg is enabled
        if self.msg:
            print(f"Starting Genetic Algorithm Optimization...")
            print(f"Population size: {sol_per_pop}")
            print(f"Generations: {num_generations}")
            print(f"Parents for mating: {num_parents_mating}")
            print(f"Links to optimize: {len(self.link_ids)}")
            if self.time_limit:
                print(f"Time limit: {self.time_limit} seconds")
            print("-" * 60)

        # Create initial population
        initial_population = self.create_initial_population(sol_per_pop)

        # Define gene space (valid range for each gene)
        gene_space = []
        for link_id in self.link_ids:
            link = self.network.links[link_id]
            num_options = len(link.get_total_capacity_options())
            gene_space.append(list(range(num_options)))

        # Create GA instance
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
            mutation_percent_genes=10,
            on_generation=self.on_generation
        )

        # Run optimization
        try:
            ga_instance.run()
        except (KeyboardInterrupt, TimeoutError):
            self.timeout_occurred = True
            if self.msg:
                print("Optimization stopped due to timeout or interruption")

        # Get the best solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        # Calculate final timing
        total_time = time.time() - self.start_time if self.start_time else 0

        # Print final results if msg is enabled
        if self.msg:
            print("-" * 60)
            print("Optimization Complete!")
            print(f"Total time: {total_time:.2f} seconds")
            print(
                f"Generations completed: {ga_instance.generations_completed}")
            print(f"Timeout occurred: {self.timeout_occurred}")
            actual_cost, penalty_cost = self.get_solution_cost(solution)
            print(f"Final best fitness: {solution_fitness:.2f}")
            print(f"Final actual cost: ${actual_cost:.2f}")
            print(f"Final penalty cost: ${penalty_cost:.2f}")

        # Return comprehensive results
        return {
            'ga_instance': ga_instance,
            'best_solution': solution,
            'best_fitness': solution_fitness,
            'best_solutions_history': self.best_solutions.copy(),
            'fitness_history': self.best_fitness_values.copy(),
            'average_fitness_history': self.average_fitness_values.copy(),
            'population_diversity_history': self.population_diversity.copy(),
            'convergence_history': self.convergence_data.copy(),
            'generation_times': self.generation_times.copy(),
            'optimizer': self,
            'timeout_occurred': self.timeout_occurred,
            'total_time': total_time,
            'generations_completed': ga_instance.generations_completed,
            'final_population_size': ga_instance.sol_per_pop,
            'solution_details': self.get_solution_details(solution)
        }

    def get_solution_cost(self, solution):
        """Calculate the actual cost of a solution (for reporting purposes)"""
        total_cost = 0.0
        total_penalty = 0.0

        for i, link_id in enumerate(self.link_ids):
            link = self.network.links[link_id]
            module_choice = int(solution[i])

            capacity_options = link.get_total_capacity_options()
            if module_choice < len(capacity_options):
                capacity, cost = capacity_options[module_choice]
                total_cost += cost

                # Calculate penalties for unmet demand
                demand = self._get_link_demand(link_id)
                if demand > capacity:
                    total_penalty += (demand - capacity) * 1000
                elif capacity == 0 and demand > 0:
                    total_penalty += demand * 5000

        return total_cost, total_penalty

    def validate_solution(self, solution) -> bool:
        """Validate that the solution satisfies the one-module-per-edge constraint.

        This constraint is automatically satisfied by the gene representation:
        - Each gene represents exactly one choice for a link
        - Gene value 0 = no module
        - Gene value 1+ = specific module index

        Returns True if valid, False otherwise.
        """
        for i, link_id in enumerate(self.link_ids):
            link = self.network.links[link_id]
            module_choice = int(solution[i])

            # Check if choice is within valid range
            num_options = len(link.get_total_capacity_options())
            if module_choice < 0 or module_choice >= num_options:
                return False

        return True

    def get_solution_details(self, solution):
        """
        Get detailed solution information in a structured format for plotting/analysis.

        Returns:
            dict: Contains structured solution details including:
                - 'link_details': List of dicts with link-specific information
                - 'total_cost': Total infrastructure cost
                - 'total_penalty': Total penalty cost  
                - 'total_demand': Total network demand
                - 'total_capacity': Total network capacity
                - 'capacity_violations': Number of links with capacity violations
                - 'unmet_demand': Total unmet demand across all links
                - 'utilization_stats': Network utilization statistics
        """
        link_details = []
        total_cost = 0.0
        total_penalty = 0.0
        total_capacity_violations = 0
        total_unmet_demand = 0.0

        # Get actual cost breakdown
        actual_cost, penalty_cost = self.get_solution_cost(solution)

        for i, link_id in enumerate(self.link_ids):
            link = self.network.links[link_id]
            module_choice = int(solution[i])

            capacity_options = link.get_total_capacity_options()
            if module_choice < len(capacity_options):
                capacity, cost = capacity_options[module_choice]
                demand = self._get_link_demand(link_id)

                # Determine module selection details
                module_info = {
                    'choice_index': module_choice,
                    'choice_description': 'No module' if module_choice == 0 else f'Module {module_choice}',
                    'added_capacity': 0.0 if module_choice == 0 else link.module_capacities[module_choice - 1],
                    'added_cost': 0.0 if module_choice == 0 else link.module_costs[module_choice - 1]
                }

                # Calculate utilization and violations
                utilization = (demand / capacity * 100) if capacity > 0 else 0
                has_violation = demand > capacity
                violation_amount = max(0, demand - capacity)

                if has_violation:
                    total_capacity_violations += 1
                    total_unmet_demand += violation_amount

                link_detail = {
                    'link_id': link_id,
                    'source': link.source,
                    'target': link.target,
                    'pre_installed_capacity': link.pre_installed_capacity,
                    'total_capacity': capacity,
                    'total_cost': cost,
                    'demand': demand,
                    'utilization_percent': utilization,
                    'has_capacity_violation': has_violation,
                    'violation_amount': violation_amount,
                    'module_info': module_info
                }

                link_details.append(link_detail)
                total_cost += cost

        # Calculate network-wide statistics
        total_demand = sum(self._get_all_link_demands().values())
        total_capacity = sum(detail['total_capacity']
                             for detail in link_details)

        # Utilization statistics
        utilizations = [detail['utilization_percent']
                        for detail in link_details if detail['total_capacity'] > 0]
        utilization_stats = {
            'mean': np.mean(utilizations) if utilizations else 0,
            'std': np.std(utilizations) if utilizations else 0,
            'min': np.min(utilizations) if utilizations else 0,
            'max': np.max(utilizations) if utilizations else 0,
            'network_overall': (total_demand / total_capacity * 100) if total_capacity > 0 else 0
        }

        return {
            'link_details': link_details,
            'total_cost': actual_cost,
            'total_penalty': penalty_cost,
            'total_demand': total_demand,
            'total_capacity': total_capacity,
            'capacity_violations': total_capacity_violations,
            'unmet_demand': total_unmet_demand,
            'utilization_stats': utilization_stats
        }


def main():
    """Main function to demonstrate the genetic algorithm"""
    results = run_genetic_algorithm(
        nodes_file="nodes.txt",
        links_file="links.txt",
        demands_file="demands.txt",
        num_generations=50,
        sol_per_pop=50,
        num_parents_mating=20,
        time_limit=10,
        msg=True,
    )

    return results


def run_genetic_algorithm(nodes_file="nodes.txt", links_file="links.txt", demands_file="demands.txt",
                          num_generations=50, sol_per_pop=50, num_parents_mating=20, time_limit=None, msg=False):
    """
    Run genetic algorithm optimization and return results for external plotting/analysis.

    Args:
        nodes_file: Path to nodes file
        links_file: Path to links file  
        demands_file: Path to demands file
        num_generations: Number of GA generations
        sol_per_pop: Population size
        num_parents_mating: Number of parents for mating
        time_limit: Maximum time in seconds (None for no limit)
        msg: Whether to show solver messages

    Returns:
        dict: Contains all optimization results including:
            - 'best_solution': The best solution found
            - 'best_fitness': Fitness of best solution
            - 'fitness_history': List of best fitness values per generation
            - 'average_fitness_history': List of average population fitness per generation
            - 'population_diversity_history': List of population diversity per generation
            - 'convergence_history': List of fitness improvements per generation
            - 'generation_times': List of cumulative time at each generation
            - 'best_solutions_history': List of best solutions per generation
            - 'optimizer': The optimizer instance for further analysis
            - 'ga_instance': The GA instance
            - 'network': The network graph
            - 'cost_history': List of actual costs per generation
            - 'penalty_history': List of penalty costs per generation
            - 'utilization_history': List of network utilization per generation
            - 'capacity_violation_history': List of capacity violations per generation
            - 'timeout_occurred': Whether optimization was stopped due to time limit
            - 'total_time': Total optimization time in seconds
            - 'generations_completed': Number of generations actually completed
            - 'solution_details': Detailed analysis of the final solution
            - 'plotting_data': Structured data specifically for creating visualizations:
                * 'generations': Generation numbers for x-axis
                * 'execution_timeline': Timing data for performance analysis
                * 'performance_metrics': Improvement and convergence metrics
                * 'solution_quality': Quality metrics over time
    """
    # Create network and load data from files
    network = NetworkGraph()
    network.load_from_files(nodes_file, links_file, demands_file)

    # Create optimizer with time limit and message settings
    optimizer = GeneticAlgorithmOptimizer(
        network, time_limit=time_limit, msg=msg)

    # Run optimization
    results = optimizer.optimize(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating
    )

    # Add network to results
    results['network'] = network

    # Calculate cost history for plotting
    cost_history = []
    penalty_history = []
    utilization_history = []
    capacity_violation_history = []

    for solution in results['best_solutions_history']:
        actual_cost, penalty_cost = optimizer.get_solution_cost(solution)
        cost_history.append(actual_cost)
        penalty_history.append(penalty_cost)

        # Get detailed solution info for additional metrics
        solution_details = optimizer.get_solution_details(solution)
        utilization_history.append(
            solution_details['utilization_stats']['network_overall'])
        capacity_violation_history.append(
            solution_details['capacity_violations'])

    results['cost_history'] = cost_history
    results['penalty_history'] = penalty_history
    results['utilization_history'] = utilization_history
    results['capacity_violation_history'] = capacity_violation_history

    # Add additional plotting data
    results['plotting_data'] = {
        'generations': list(range(1, len(results['fitness_history']) + 1)),
        'execution_timeline': {
            'generation_times': results['generation_times'],
            'cumulative_times': [sum(results['generation_times'][:i+1]) for i in range(len(results['generation_times']))],
            'time_per_generation': [results['generation_times'][i] - (results['generation_times'][i-1] if i > 0 else 0) for i in range(len(results['generation_times']))]
        },
        'performance_metrics': {
            'fitness_improvement': [(results['fitness_history'][i] - results['fitness_history'][0]) for i in range(len(results['fitness_history']))],
            'cost_reduction': [(cost_history[0] - cost_history[i]) for i in range(len(cost_history))],
            'convergence_rate': results['convergence_history']
        },
        'solution_quality': {
            'best_fitness_per_generation': results['fitness_history'],
            'average_fitness_per_generation': results['average_fitness_history'],
            'population_diversity_per_generation': results['population_diversity_history'],
            'network_utilization_per_generation': utilization_history,
            'capacity_violations_per_generation': capacity_violation_history
        }
    }

    return results


if __name__ == "__main__":
    results = main()
