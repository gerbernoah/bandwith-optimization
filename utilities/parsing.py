"""
Parsing functions for input files.

functions: parse_nodes, parse_demands, parse_links
"""
# this works when this code is run from root directory
from types.network import Node, Demand, Link, Module, NodeDict

# normal imports
from typing import List, Dict, Tuple
import re
from pathlib import Path

"""
==================================================
 PARSING FUNCTIONS
==================================================
"""

# Define input file paths; files located in "../input" folder
current_dir = Path.cwd()
input_dir = current_dir.parent / "input"
input_file = lambda file: str(input_dir / file)

def parse_nodes(file_path: str) -> Tuple[List[Node], NodeDict]:
    nodes: List[Node] = []
    node_dict: NodeDict = {}

    # open file and iterate through lines
    with open(input_file("nodes.txt"), 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:  # Skip header
        line = line.strip()
        parts = line.split('(')
        if len(parts) < 2:
            continue

        # Extract: name, create Node, map name to Node object
        name = parts[0].strip()

        # store Node
        node = Node(name=name)
        nodes.append(node)
        node_dict[name] = node
    return nodes, node_dict

def parse_demands(node_dict: NodeDict) -> List[Demand]:
    demands: List[Demand] = []

    # open file and iterate through lines
    with open(input_file("demands.txt"), 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:  # Skip header
        line = line.strip()
        parts = line.split()
        if len(parts) < 4:
            continue

        # Extract: source, target, demand_value
        source = node_dict.get(parts[1])
        target = node_dict.get(parts[2])
        demand_value = float(parts[3])

        # store Demand
        if source is not None and target is not None:
            demand = Demand(source=source, target=target, demand_value=demand_value)
            demands.append(demand)
    return demands

def parse_links(node_dict: NodeDict) -> List[Link]:
    links: List[Link] = []

    # open file and iterate through lines
    with open(input_file("links.txt"), 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:  # Skip header
        line = line.strip()

        # note: parentheses without backslash represent a group in regex
        # \S+ = one or more non-whitespace characters
        # \d+ = one or more digits
        # \s* = zero or more whitespace characters
        pattern = re.compile(
            r"(\S+)\s+"                             # link_id                           | group 1
            r"\(\s*(\S+)\s+(\S+)\s*\)\s+"           # ( source target )                 | groups 2, 3
            r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"     # number number routing_cost number | groups 4, 5, 6, 7
            r"\(\s*((?:\d+\s+\d+\s*)*)\)"           # ( {module_capacity module_cost}* )| group 8    
        )

        match = pattern.match(line)
        if not match:
            raise ValueError("Line format is incorrect")

        # Extract: source, target, routing_cost, modules
        source, target, route_cost = match.group(2, 3, 6)
        modules_raw = match.group(8)
        modules = re.findall(r"(\d+)\s+(\d+)", modules_raw) # Extract modules from string of group 8
        modules = [Module(cap, cost) for cap, cost in modules]

        # store Link
        link = Link(
            source = node_dict[source],
            target = node_dict[target],
            routing_cost = float(route_cost),
            modules = modules
        )
        links.append(link)
        
    return links
