import importlib
import pulp
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
network_module = importlib.import_module('network')
NetworkGraph = network_module.NetworkGraph


class IntegerProgrammingOptimizer:
    """Integer Programming implementation for network optimization"""

    def __init__(self, network: NetworkGraph, time_limit=None):
        self.network = network
        self.link_ids = list(network.links.keys())
        self.time_limit = time_limit
        self.start_time = None
        self.solve_time = 0

        # Prepare data structures for PuLP using network.py functionality
        self.nodeids = {node_id: i for i,
                        node_id in enumerate(network.nodes.keys())}
        self.V = list(range(len(network.nodes)))
        self.E = self._create_directed_edges()
        self.M = self._create_modules_dict_from_network()
        self.D = self._create_demands_from_network()

    def _create_directed_edges(self):
        """Create directed edges from undirected network links"""
        E = {}
        for link_id, link in self.network.links.items():
            src_idx = self.nodeids[link.source]
            tgt_idx = self.nodeids[link.target]

            # Forward direction
            E[f"{link_id}_fwd"] = (src_idx, tgt_idx, link.routing_cost)
            # Reverse direction
            E[f"{link_id}_rev"] = (tgt_idx, src_idx, link.routing_cost)

        return E

    def _create_modules_dict_from_network(self):
        """Create modules dictionary using the network.py Link.get_total_capacity_options() method"""
        M = {}
        for link_id, link in self.network.links.items():
            # Use the existing method from network.py Link class
            M[link_id] = link.get_total_capacity_options()
        return M

    def _create_demands_from_network(self):
        """Create demands list using network.py Demand objects"""
        D = []
        for demand_id, demand in self.network.demands.items():
            src_idx = self.nodeids[demand.source]
            tgt_idx = self.nodeids[demand.target]
            D.append((src_idx, tgt_idx, demand.demand_value))
        return D

    def optimize(self, solver_name="CPLEX_PY", msg=False):
        """
        Run the integer programming optimization using direct solver model modification

        Args:
            solver_name: PuLP solver name to use
            msg: Whether to show solver messages

        Returns:
            dict: Comprehensive results similar to genetic algorithm
        """
        self.start_time = time.time()

        # Initialize problem
        prob = pulp.LpProblem("NetworkDesign", pulp.LpMinimize)

        # Decision variables
        x = {}  # flow variables: x[e][s][t]
        y = {}  # module installation: y[e][m]

        # Create flow variables for each directed edge
        for e in self.E:
            x[e] = {}
            for s, t, d in self.D:
                x[e][(s, t)] = pulp.LpVariable(f"x_{e}_{s}_{t}", lowBound=0)

        # Create module variables for each base (undirected) edge
        for base_id in self.M:
            y[base_id] = {}
            for idx, (cap, cost) in enumerate(self.M[base_id]):
                y[base_id][idx] = pulp.LpVariable(
                    f"y_{base_id}_{idx}", cat="Binary")

        # Objective: minimize routing cost + module cost
        routing_cost = pulp.lpSum(
            x[e][(s, t)] * self.E[e][2]
            for e in self.E
            for (s, t, _) in self.D
        )
        module_cost = pulp.lpSum(
            y[e][m] * self.M[e][m][1]
            for e in self.M
            for m in y[e]
        )
        prob += routing_cost + module_cost

        # Flow conservation constraints
        for s, t, d_val in self.D:
            for v in self.V:
                inflow = pulp.lpSum(
                    x[e][(s, t)] for e in self.E if self.E[e][1] == v
                )
                outflow = pulp.lpSum(
                    x[e][(s, t)] for e in self.E if self.E[e][0] == v
                )

                if v == s:
                    prob += (outflow - inflow ==
                             d_val), f"flow_src_{s}_{t}_{v}"
                elif v == t:
                    prob += (inflow - outflow ==
                             d_val), f"flow_tgt_{s}_{t}_{v}"
                else:
                    prob += (inflow == outflow), f"flow_bal_{s}_{t}_{v}"

        # Capacity constraints
        for base_edge_id in self.M:
            total_flow_on_edge = pulp.lpSum(
                x[f"{base_edge_id}_fwd"][(s, t)] +
                x[f"{base_edge_id}_rev"][(s, t)]
                for (s, t, _) in self.D
            )
            prob += (
                total_flow_on_edge <=
                pulp.lpSum(y[base_edge_id][m] * self.M[base_edge_id][m][0]
                           for m in y[base_edge_id])
            ), f"cap_{base_edge_id}"

        # One module per edge constraint
        for base_edge_id in self.M:
            prob += (
                pulp.lpSum(y[base_edge_id][m] for m in y[base_edge_id]) == 1
            ), f"one_module_{base_edge_id}"

        # Solve the problem using direct solver model modification where possible
        try:
            if solver_name == "CPLEX_PY":
                solver = pulp.CPLEX_PY(msg=msg)
                solver.buildSolverModel(prob)

                # Set time limit if specified
                if self.time_limit and self.time_limit > 0:
                    solver.solverModel.parameters.timelimit.set(
                        self.time_limit)

                # Solve using direct solver model
                solver.callSolver(prob)
                status = solver.findSolutionValues(prob)

            elif solver_name == "GUROBI_PY":
                solver = pulp.GUROBI_PY(msg=msg)
                solver.buildSolverModel(prob)

                # Set time limit if specified
                if self.time_limit and self.time_limit > 0:
                    solver.solverModel.setParam('TimeLimit', self.time_limit)

                # Solve using direct solver model
                solver.callSolver(prob)
                status = solver.findSolutionValues(prob)

            else:
                # Fall back to traditional solving for other solvers (including CBC)
                if solver_name == "PULP_CBC_CMD":
                    if self.time_limit and self.time_limit > 0:
                        options = [f"seconds {self.time_limit}"]
                        solver = pulp.PULP_CBC_CMD(msg=msg, options=options)
                    else:
                        solver = pulp.PULP_CBC_CMD(msg=msg)
                else:
                    solver = pulp.getSolver(solver_name, msg=msg)

                # Traditional solve
                status = prob.solve(solver)

        except KeyboardInterrupt:
            print("🛑 Optimization interrupted by user")
            status = pulp.LpStatusNotSolved
        except Exception as e:
            print(f"❌ Solver error: {e}")
            print("   Falling back to traditional solving...")

            # Fall back to traditional solving
            try:
                if solver_name == "PULP_CBC_CMD":
                    if self.time_limit and self.time_limit > 0:
                        options = [f"seconds {self.time_limit}"]
                        solver = pulp.PULP_CBC_CMD(msg=msg, options=options)
                    else:
                        solver = pulp.PULP_CBC_CMD(msg=msg)
                else:
                    solver = pulp.getSolver(solver_name, msg=msg)

                status = prob.solve(solver)
            except KeyboardInterrupt:
                print("🛑 Fallback solving interrupted by user")
                status = pulp.LpStatusNotSolved
            except Exception as fallback_error:
                print(f"❌ Fallback solver also failed: {fallback_error}")
                status = pulp.LpStatusInfeasible

        # Calculate solve time
        self.solve_time = time.time() - self.start_time if self.start_time else 0

        # Extract solution and objective value
        solution = None
        objective_value = None

        if status == pulp.LpStatusOptimal:
            try:
                solution = self._extract_solution(y)
                objective_value = pulp.value(prob.objective)
            except Exception as e:
                print(f"Error extracting optimal solution: {e}")
                solution = None
                objective_value = None
        else:
            print(
                f"Solver finished with status: {pulp.LpStatus[status]} - no solution available")
            solution = None
            objective_value = float('inf')

        results = {
            'best_solution': solution,
            'best_fitness': objective_value if objective_value is not None else float('inf'),
            'optimizer': self,
            'total_time': self.solve_time,
            'status': pulp.LpStatus[status],
            'problem': prob,
            'variables': {'x': x, 'y': y}
        }

        # Add solution details if optimal solution found
        if solution is not None:
            results['solution_details'] = self.get_solution_details(solution)
        else:
            results['solution_details'] = self._get_empty_solution_details()

        return results

    def _extract_solution(self, y):
        """Extract solution vector from optimization variables"""
        solution = []
        for link_id in self.link_ids:
            # Find which module was selected
            selected_module = 0
            found_selection = False

            try:
                for m in y[link_id]:
                    var_value = pulp.value(y[link_id][m])
                    if var_value is not None and var_value > 0.5:
                        selected_module = m
                        found_selection = True
                        break

                # If no module was clearly selected, check if this is a partial solution
                if not found_selection:
                    # For partial solutions (timeouts), default to module 0 if no selection found
                    # Check if all variables are None (indicating completely incomplete solve)
                    all_none = all(pulp.value(
                        y[link_id][m]) is None for m in y[link_id])
                    if all_none:
                        # This might be a timeout case - use default module 0
                        selected_module = 0
                    else:
                        # Some variables have values but none > 0.5, still use module 0
                        selected_module = 0

            except Exception as e:
                # For any error in variable access, default to module 0
                selected_module = 0

            solution.append(selected_module)
        return solution

    def get_solution_details(self, solution):
        """Get detailed solution information similar to genetic algorithm"""
        if solution is None:
            return self._get_empty_solution_details()

        link_details = []
        total_cost = 0.0
        total_capacity_violations = 0
        total_unmet_demand = 0.0

        for i, link_id in enumerate(self.link_ids):
            link = self.network.links[link_id]
            module_choice = int(solution[i])

            # Get capacity and cost for selected module using network.py functionality
            capacity_options = link.get_total_capacity_options()
            if module_choice < len(capacity_options):
                capacity, cost = capacity_options[module_choice]
                demand = self._get_link_demand(link_id)

                # Module selection details
                if module_choice == 0:
                    module_info = {
                        'choice_index': module_choice,
                        'choice_description': 'No module',
                        'added_capacity': 0.0,
                        'added_cost': 0.0,
                        'total_cost': cost
                    }
                else:
                    module_info = {
                        'choice_index': module_choice,
                        'choice_description': f'Module {module_choice}',
                        'added_capacity': capacity - link.pre_installed_capacity,
                        'added_cost': cost - link.pre_installed_capacity_cost,
                        'total_cost': cost
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

        # Calculate network statistics
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
            'total_cost': total_cost,
            'total_penalty': 0.0,  # IP doesn't use penalties like GA
            'total_demand': total_demand,
            'total_capacity': total_capacity,
            'capacity_violations': total_capacity_violations,
            'unmet_demand': total_unmet_demand,
            'utilization_stats': utilization_stats
        }

    def _get_empty_solution_details(self):
        """Return empty solution details when no solution found"""
        return {
            'link_details': [],
            'total_cost': float('inf'),
            'total_penalty': 0.0,
            'total_demand': 0.0,
            'total_capacity': 0.0,
            'capacity_violations': 0,
            'unmet_demand': 0.0,
            'utilization_stats': {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'network_overall': 0
            }
        }

    def _get_link_demand(self, link_id: str) -> float:
        """
        Calculate traffic demand for a specific link by finding all demands
        that could potentially use this link (between the link's endpoints)
        """
        link = self.network.links[link_id]
        total_demand = 0.0

        # Find all demands between the link's endpoints (bidirectional)
        for demand in self.network.demands.values():
            if ((demand.source == link.source and demand.target == link.target) or
                    (demand.source == link.target and demand.target == link.source)):
                total_demand += demand.demand_value

        return total_demand

    def _get_all_link_demands(self):
        """Get traffic demands for all links using network demand objects"""
        demands = {}
        for link_id in self.link_ids:
            demands[link_id] = self._get_link_demand(link_id)
        return demands

    def _calculate_objective_from_solution(self, solution):
        """Calculate objective value from solution when solver doesn't provide it"""
        total_cost = 0.0

        for i, link_id in enumerate(self.link_ids):
            link = self.network.links[link_id]
            module_choice = int(solution[i])

            # Get cost for selected module
            capacity_options = link.get_total_capacity_options()
            if module_choice < len(capacity_options):
                _, cost = capacity_options[module_choice]
                total_cost += cost

        return total_cost


def run_integer_programming(nodes_file="nodes.txt", links_file="links.txt", demands_file="demands.txt",
                            solver_name="PULP_CBC_CMD", time_limit=None, msg=False):
    """
    Run integer programming optimization and return results similar to genetic algorithm.

    Args:
        nodes_file: Path to nodes file
        links_file: Path to links file  
        demands_file: Path to demands file
        solver_name: PuLP solver name
        time_limit: Maximum time in seconds (None for no limit)
        msg: Whether to show solver messages

    Returns:
        dict: Contains all optimization results including:
            - 'best_solution': The optimal solution found
            - 'best_fitness': Objective value of best solution  
            - 'optimizer': The optimizer instance
            - 'network': The network graph
            - 'solution_details': Detailed analysis of the solution
            - 'total_time': Total optimization time in seconds
            - 'status': Solver status
            - 'plotting_data': Structured data for visualization (simplified for IP)
    """
    # Create network and load data using network.py functionality
    network = NetworkGraph()
    network.load_from_files(nodes_file, links_file, demands_file)

    # Create optimizer
    optimizer = IntegerProgrammingOptimizer(
        network, time_limit=time_limit)

    # Run optimization with interrupt handling
    try:
        results = optimizer.optimize(solver_name=solver_name, msg=msg)
    except KeyboardInterrupt:
        print("\n🛑 Integer programming optimization interrupted by user")
        # Create minimal results structure for interrupted case
        results = {
            'best_solution': None,
            'best_fitness': float('inf'),
            'optimizer': optimizer,
            'total_time': optimizer.solve_time if hasattr(optimizer, 'solve_time') else 0.0,
            'status': 'Interrupted',
            'problem': None,
            'variables': {'x': {}, 'y': {}},
            'solution_details': optimizer._get_empty_solution_details()
        }

    # Add network to results
    results['network'] = network

    # Create simplified plotting data for consistency with genetic algorithm
    results['plotting_data'] = {
        'generations': [1],  # IP is single-shot, so only one "generation"
        'execution_timeline': {
            'generation_times': [results['total_time']],
            'cumulative_times': [results['total_time']],
            'time_per_generation': [results['total_time']]
        },
        'performance_metrics': {
            # No improvement over generations for IP
            'fitness_improvement': [0],
            'cost_reduction': [0],
            'convergence_rate': [0]
        },
        'solution_quality': {
            'best_fitness_per_generation': [results['best_fitness']],
            # Same as best for IP
            'average_fitness_per_generation': [results['best_fitness']],
            # No population diversity for IP
            'population_diversity_per_generation': [0],
            'network_utilization_per_generation': [results['solution_details']['utilization_stats']['network_overall']],
            'capacity_violations_per_generation': [results['solution_details']['capacity_violations']]
        }
    }

    # Add single-point histories for compatibility
    results['fitness_history'] = [results['best_fitness']]
    results['cost_history'] = [results['solution_details']['total_cost']]
    results['penalty_history'] = [results['solution_details']['total_penalty']]
    results['utilization_history'] = [results['solution_details']
                                      ['utilization_stats']['network_overall']]
    results['capacity_violation_history'] = [
        results['solution_details']['capacity_violations']]
    results['generation_times'] = [results['total_time']]
    results['generations_completed'] = 1

    return results


def main():
    """Main function to demonstrate integer programming optimization"""
    print("Running Integer Programming Optimization...")

    # Use relative paths from the integer_programming directory
    results = run_integer_programming(
        nodes_file="../nodes.txt",
        links_file="../links.txt",
        demands_file="../demands.txt",
        time_limit=20,  # 2 seconds time limit for testing
        msg=True
    )

    print(f"\nOptimization Results:")
    print(f"Status: {results['status']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Best objective: {results['best_fitness']:.2f}")

    solution_details = results['solution_details']
    print(f"Total cost: ${solution_details['total_cost']:.2f}")
    print(
        f"Network utilization: {solution_details['utilization_stats']['network_overall']:.1f}%")
    print(f"Capacity violations: {solution_details['capacity_violations']}")

    return results


if __name__ == "__main__":
    results = main()
