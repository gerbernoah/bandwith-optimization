"""
Parsing functions for input files.

functions: parse_nodes, parse_demands, parse_edges, parse_network
"""
# this works when this code is run from root directory
from types2.network import Node, Demand, Edge, Module, NodeDict, UEdge, UEdgeToEdge, Network

# normal imports
from typing import List, Dict, Tuple
import re
from pathlib import Path

"""
==================================================
 PARSING FUNCTIONS
==================================================
"""

# Define input file paths; files located in "../inputs" folder
input_dir = Path(__file__).resolve().parent.parent / "inputs"
input_file = lambda file: str(input_dir / file)

def parse_nodes() -> Tuple[List[Node], NodeDict]:
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

        # Extract: source, target, value
        source = node_dict.get(parts[2])
        target = node_dict.get(parts[3])
        value = float(parts[6])

        # store Demand
        if source is not None and target is not None:
            demand = Demand(source=source, target=target, value=value)
            demands.append(demand)
    return demands

def parse_edges(node_dict: NodeDict) -> Tuple[List[Edge], List[UEdge], UEdgeToEdge]:
    edges: List[Edge] = []
    uedges: List[UEdge] = []
    uedge_to_edge: UEdgeToEdge = {}

    # open file and iterate through lines
    with open(input_file("edges.txt"), 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:  # Skip header
        line = line.strip()

        # note: parentheses without backslash represent a group in regex
        # \S+ = one or more non-whitespace characters
        # \d+ = one or more digits
        # \s* = zero or more whitespace characters
        pattern = re.compile(
            r"(\S+)\s+"                                         # edge_id                                           | group 1
            r"\(\s*(\S+)\s+(\S+)\s*\)\s+"                       # ( source target )                                 | groups 2, 3
            r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"     # floats: pre_cap, pre_cost, route_cost, setup_cost | groups 4–7
            r"\(\s*([\d.\s]+)\)"                                # flat list of floats                               | group 8
        )


        match = pattern.match(line)
        if not match:
            raise ValueError("Line format is incorrect")

        # Extract: source, target, routing_cost, modules
        source, target, route_cost = match.group(2, 3, 6)
        modules_raw = match.group(8)
        moduleGroups = re.findall(r"(\d+\.\d+)\s+(\d+\.\d+)", modules_raw) # Extract modules from string of group 8
        modules: List[Module] = [Module(capacity=float(cap), cost=float(cost), index=index) for index, (cap, cost) in enumerate(moduleGroups)]

        # create undirected UEdge and two directed Edge objects
        uEdge = UEdge(routing_cost=float(route_cost), module_options=modules)
        edge1 = Edge(
            source = node_dict[source],
            target = node_dict[target],
            uEdge = uEdge
        )
        edge2 = Edge(
            source = node_dict[target],
            target = node_dict[source],
            uEdge = uEdge
        )
        edges.append(edge1)
        edges.append(edge2)
        uedges.append(uEdge)
        uedge_to_edge[uEdge.id] = (edge1, edge2)

    return edges, uedges, uedge_to_edge

def parse_network() -> Network:
    nodes, node_dict = parse_nodes()
    edges, uedges, uedge_to_edge = parse_edges(node_dict)
    demands= parse_demands(node_dict)

    return Network(
        nodes = nodes,
        node_dict = node_dict,
        edges = edges,
        uedges = uedges,
        uedge_to_edge = uedge_to_edge,
        demands = demands
    )