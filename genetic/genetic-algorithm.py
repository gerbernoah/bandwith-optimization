import pygad
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import re
import os


@dataclass
class Link:
    """Represents a network link with capacity and cost information"""
    link_id: str
    source: str
    target: str
    pre_installed_capacity: float
    pre_installed_capacity_cost: float
    routing_cost: float
    setup_cost: float
    module_capacities: List[float]
    module_costs: List[float]

    def get_total_capacity_options(self) -> List[Tuple[float, float]]:
        """Returns list of (total_capacity, total_cost) options"""
        options = [(self.pre_installed_capacity,
                    self.pre_installed_capacity_cost)]

        for i, (capacity, cost) in enumerate(zip(self.module_capacities, self.module_costs)):
            total_capacity = self.pre_installed_capacity + capacity
            total_cost = self.pre_installed_capacity_cost + cost + self.setup_cost
            options.append((total_capacity, total_cost))

        return options


@dataclass
class Node:
    """Represents a network node with coordinates"""
    node_id: str
    longitude: float
    latitude: float


@dataclass
class Demand:
    """Represents a traffic demand between two nodes"""
    demand_id: str
    source: str
    target: str
    routing_unit: int
    demand_value: float
    max_path_length: str  # "UNLIMITED" or numeric value


class NetworkGraph:
    """Represents the network graph and handles optimization"""

    def __init__(self):
        self.links: Dict[str, Link] = {}
        self.nodes: Dict[str, Node] = {}
        self.demands: Dict[str, Demand] = {}

    def load_from_files(self, nodes_file="nodes.txt", links_file="links.txt", demands_file="demands.txt"):
        """Load network data from external files"""
        if os.path.exists(nodes_file):
            self.parse_nodes_from_file(nodes_file)
        if os.path.exists(links_file):
            self.parse_links_from_file(links_file)
        if os.path.exists(demands_file):
            self.parse_demands_from_file(demands_file)

    def parse_nodes_from_file(self, filename: str):
        """Parse nodes from external file"""
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header
            line = line.strip()
            if line and not line.startswith('<'):
                self._parse_single_node(line)

    def _parse_single_node(self, line: str):
        """Parse a single node line"""
        # Format: NodeName ( longitude latitude )
        parts = line.split('(')
        if len(parts) < 2:
            return

        node_id = parts[0].strip()
        coords_part = parts[1].rstrip(')').strip()
        coords = coords_part.split()

        if len(coords) >= 2:
            longitude = float(coords[0])
            latitude = float(coords[1])

            node = Node(
                node_id=node_id,
                longitude=longitude,
                latitude=latitude
            )
            self.nodes[node_id] = node

    def parse_demands_from_file(self, filename: str):
        """Parse demands from external file"""
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header
            line = line.strip()
            if line and not line.startswith('<'):
                self._parse_single_demand(line)

    def _parse_single_demand(self, line: str):
        """Parse a single demand line"""
        # Format: D1 ( Amsterdam Athens ) 1 179.00 UNLIMITED
        parts = line.split()
        if len(parts) < 6:
            return

        demand_id = parts[0]
        source = parts[2]
        target = parts[3]
        routing_unit = int(parts[5])
        demand_value = float(parts[6])
        max_path_length = parts[7]

        demand = Demand(
            demand_id=demand_id,
            source=source,
            target=target,
            routing_unit=routing_unit,
            demand_value=demand_value,
            max_path_length=max_path_length
        )
        self.demands[demand_id] = demand

    def parse_links_from_file(self, filename: str):
        """Parse links from external file"""
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header
            line = line.strip()
            if line and not line.startswith('<'):
                self._parse_single_link(line)

    def _parse_single_link(self, line: str):
        """Parse a single link line"""
        # Extract link_id
        link_id = line.split('(')[0].strip()

        # Extract source and target
        nodes_match = re.search(r'\(\s*(\w+)\s+(\w+)\s*\)', line)
        if not nodes_match:
            return

        source, target = nodes_match.groups()

        # Extract numerical values
        # Find everything after the nodes parentheses
        after_nodes = line.split(')', 1)[1].strip()

        # Split by parentheses to separate basic values from module data
        parts = after_nodes.split('(', 1)
        basic_values = parts[0].strip().split()

        pre_installed_capacity = float(basic_values[0])
        pre_installed_capacity_cost = float(basic_values[1])
        routing_cost = float(basic_values[2])
        setup_cost = float(basic_values[3])

        # Extract module data
        module_capacities = []
        module_costs = []

        if len(parts) > 1:
            module_data = parts[1].rstrip(')').strip()
            module_values = module_data.split()

            # Pair up values (capacity, cost)
            for i in range(0, len(module_values), 2):
                if i + 1 < len(module_values):
                    module_capacities.append(float(module_values[i]))
                    module_costs.append(float(module_values[i + 1]))

        link = Link(
            link_id=link_id,
            source=source,
            target=target,
            pre_installed_capacity=pre_installed_capacity,
            pre_installed_capacity_cost=pre_installed_capacity_cost,
            routing_cost=routing_cost,
            setup_cost=setup_cost,
            module_capacities=module_capacities,
            module_costs=module_costs
        )

        self.links[link_id] = link

    def add_sample_links(self):
        """Add the sample links from your example"""
        sample_text = """
        L1 ( Amsterdam Brussels ) 0.00 0.00 2.59 0.00 ( 7560.00 23310.00 30240.00 69930.00 120960.00 209790.00 )
        L2 ( Amsterdam Glasgow ) 0.00 0.00 10.67 0.00 ( 7560.00 96030.00 30240.00 288090.00 120960.00 864270.00 )
        L3 ( Amsterdam Hamburg ) 0.00 0.00 5.52 0.00 ( 7560.00 49680.00 30240.00 149040.00 120960.00 447120.00 )
        """
        self.parse_links_from_text(sample_text)


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm implementation for network optimization"""

    def __init__(self, network: NetworkGraph):
        self.network = network
        self.link_ids = list(network.links.keys())
        self.num_genes = len(self.link_ids)

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

    def create_initial_population(self, sol_per_pop: int) -> np.ndarray:
        """Create initial population where each gene represents module choice for a link"""
        population = []

        for _ in range(sol_per_pop):
            solution = []
            for link_id in self.link_ids:
                link = self.network.links[link_id]
                num_options = len(link.get_total_capacity_options())
                # Random choice of module option (0 = no module, 1+ = module index)
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
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        self.best_solutions.append(solution.copy())
        self.best_fitness_values.append(solution_fitness)

        # Calculate actual cost for this generation
        actual_cost, penalty_cost = self.get_solution_cost(solution)
        total_cost = actual_cost + penalty_cost

        print(f"Generation {ga_instance.generations_completed}: "
              f"Fitness = {solution_fitness:.2f}, "
              f"Cost = {actual_cost:.2f}, "
              f"Penalties = {penalty_cost:.2f}")

        # Show best cost improvement every 10 generations
        if ga_instance.generations_completed % 10 == 0:
            print(f"  → Best Total Cost: {total_cost:.2f}")

    def optimize(self, num_generations=100, sol_per_pop=50, num_parents_mating=25):
        """Run the genetic algorithm optimization"""
        print("Starting Genetic Algorithm Optimization...")
        print(f"Network has {len(self.network.links)} links")
        print(f"Links: {list(self.network.links.keys())}")

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
        ga_instance.run()

        return ga_instance

    def analyze_solution(self, solution):
        """Analyze and print details of a solution"""
        print("\n=== Solution Analysis ===")
        total_cost = 0.0
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

                print(f"Link {link_id} ({link.source} -> {link.target}):")
                print(f"  Module choice: {module_choice}")
                print(f"  Capacity: {capacity:.2f}")
                print(f"  Cost: {cost:.2f}")
                print(f"  Demand: {demand:.2f}")

                if capacity > 0:
                    utilization = (demand / capacity * 100)
                    print(f"  Utilization: {utilization:.1f}%")
                    if demand > capacity:
                        violation = demand - capacity
                        print(
                            f"  ⚠️  CAPACITY VIOLATION: {violation:.2f} units over capacity")
                        total_capacity_violations += 1
                        total_unmet_demand += violation
                else:
                    print(f"  Utilization: N/A (no capacity)")
                    if demand > 0:
                        print(
                            f"  ⚠️  UNMET DEMAND: {demand:.2f} units with no capacity")
                        total_capacity_violations += 1
                        total_unmet_demand += demand
                print()

                total_cost += cost

        print(f"=== Cost Summary ===")
        print(f"Total Infrastructure Cost: {actual_cost:.2f}")
        print(f"Total Penalty Cost: {penalty_cost:.2f}")
        print(f"Combined Cost: {actual_cost + penalty_cost:.2f}")
        print(f"Capacity Violations: {total_capacity_violations}")
        print(f"Total Unmet Demand: {total_unmet_demand:.2f}")

        # Calculate network-wide statistics
        total_demand = sum(self._get_all_link_demands().values())
        total_capacity = sum(
            link.get_total_capacity_options()[int(solution[i])][0]
            for i, link in enumerate(self.network.links.values())
        )

        print(f"\n=== Network Statistics ===")
        print(f"Network Total Demand: {total_demand:.2f}")
        print(f"Network Total Capacity: {total_capacity:.2f}")
        if total_capacity > 0:
            print(
                f"Network Utilization: {(total_demand/total_capacity*100):.1f}%")

    def plot_convergence(self):
        """Plot the convergence of the genetic algorithm"""
        plt.figure(figsize=(12, 8))

        # Create subplot for fitness convergence
        plt.subplot(2, 1, 1)
        plt.plot(self.best_fitness_values, 'b-', linewidth=2)
        plt.title('Genetic Algorithm Convergence - Fitness Values')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)

        # Create subplot for cost convergence (derived from fitness)
        plt.subplot(2, 1, 2)
        costs = []
        for i, solution in enumerate(self.best_solutions):
            actual_cost, penalty_cost = self.get_solution_cost(solution)
            costs.append(actual_cost + penalty_cost)

        plt.plot(costs, 'r-', linewidth=2)
        plt.title('Genetic Algorithm Convergence - Total Cost')
        plt.xlabel('Generation')
        plt.ylabel('Total Cost')
        plt.grid(True)

        plt.tight_layout()

        # Save the plot to a file
        plot_filename = 'genetic_algorithm_convergence.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved as: {plot_filename}")

        # Also try to show it (will work in some environments)
        try:
            plt.show()
        except:
            pass

        plt.close()

    def print_network_summary(self):
        """Print a summary of the network data"""
        print("\n=== Network Summary ===")
        print(f"Nodes: {len(self.network.nodes)}")
        print(f"Links: {len(self.network.links)}")
        print(f"Demands: {len(self.network.demands)}")

        # Show sample demands
        print("\nSample demands:")
        for i, (demand_id, demand) in enumerate(list(self.network.demands.items())[:5]):
            print(
                f"  {demand_id}: {demand.source} -> {demand.target}, {demand.demand_value:.0f} units")

        # Show links with highest demand
        link_demands = self._get_all_link_demands()
        sorted_links = sorted(link_demands.items(),
                              key=lambda x: x[1], reverse=True)
        print(f"\nLinks with highest demand:")
        for link_id, demand in sorted_links[:5]:
            link = self.network.links[link_id]
            print(
                f"  {link_id}: {link.source} -> {link.target}, {demand:.0f} units")

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


def main():
    """Main function to demonstrate the genetic algorithm"""
    # Create network and load data from files
    network = NetworkGraph()
    network.load_from_files()

    print(f"Loaded {len(network.nodes)} nodes")
    print(f"Loaded {len(network.links)} links")
    print(f"Loaded {len(network.demands)} demands")

    # Create optimizer
    optimizer = GeneticAlgorithmOptimizer(network)

    # Print network summary
    optimizer.print_network_summary()

    # Run optimization with improved parameters
    ga_instance = optimizer.optimize(
        num_generations=50,  # More generations for better convergence
        sol_per_pop=50,      # Larger population
        num_parents_mating=20
    )

    # Get best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print(f"\nOptimization completed!")
    print(f"Best fitness: {solution_fitness:.2f}")

    # Analyze the best solution
    optimizer.analyze_solution(solution)

    # Plot convergence
    try:
        optimizer.plot_convergence()
    except Exception as e:
        print(f"Could not display plot: {e}")

    return optimizer, ga_instance


if __name__ == "__main__":
    optimizer, ga_instance = main()
