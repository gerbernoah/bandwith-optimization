"""
Network representation and file parsing utilities.

This module contains classes for representing network components (nodes, links, demands)
and functionality for parsing network data from files.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
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
        """Returns list of (total_capacity, total_cost) options where only one module can be selected"""
        # Option 0: No additional modules, just pre-installed capacity
        options = [(self.pre_installed_capacity,
                    self.pre_installed_capacity_cost)]

        # Options 1+: Each represents selecting exactly one module
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
    """Represents the network graph and handles file parsing"""

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
