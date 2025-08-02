"""
Parsing functions for input files.

functions: parse_nodes, parse_demands, parse_edges, parse_network
"""
# this works when this code is run from root directory
from types2.network import Node, Demand, Edge, Module, NodeDict, UEdge, UEdgeToEdge, Network
from types2.biology import GAParams

# normal imports
from typing import List, Dict, Tuple, Union
import re
from pathlib import Path
<<<<<<< HEAD
import json, os
=======
import json
import os
>>>>>>> GA_optimization

"""
==================================================
 INPUT
==================================================
"""

# Define input file paths; files located in "../inputs" folder
input_dir = Path(__file__).resolve().parent.parent / "inputs"
def input_file(file): return str(input_dir / file)


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
            # edge_id                                           | group 1
            r"(\S+)\s+"
            # ( source target )                                 | groups 2, 3
            r"\(\s*(\S+)\s+(\S+)\s*\)\s+"
            # floats: pre_cap, pre_cost, route_cost, setup_cost | groups 4–7
            r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
            # flat list of floats                               | group 8
            r"\(\s*([\d.\s]+)\)"
        )

        match = pattern.match(line)
        if not match:
            raise ValueError("Line format is incorrect")

        # Extract: source, target, routing_cost, modules
        source, target, route_cost = match.group(2, 3, 6)
        modules_raw = match.group(8)
        # Extract modules from string of group 8
        moduleGroups = re.findall(r"(\d+\.\d+)\s+(\d+\.\d+)", modules_raw)
        modules: List[Module] = [Module(capacity=float(cap), cost=float(
            cost), index=index) for index, (cap, cost) in enumerate(moduleGroups)]

        # create undirected UEdge and two directed Edge objects
        uEdge = UEdge(routing_cost=float(route_cost), module_options=modules)
        edge1 = Edge(
            source=node_dict[source],
            target=node_dict[target],
            uEdge=uEdge
        )
        edge2 = Edge(
            source=node_dict[target],
            target=node_dict[source],
            uEdge=uEdge
        )
        edges.append(edge1)
        edges.append(edge2)
        uedges.append(uEdge)
        uedge_to_edge[uEdge.id] = (edge1, edge2)

    return edges, uedges, uedge_to_edge


def parse_network() -> Network:
    nodes, node_dict = parse_nodes()
    edges, uedges, uedge_to_edge = parse_edges(node_dict)
    demands = parse_demands(node_dict)

    return Network(
<<<<<<< HEAD
        nodes = nodes,
        node_dict = node_dict,
        edges = edges,
        uedges = uedges,
        uedge_to_edge = uedge_to_edge,
        demands = demands
    )

=======
        nodes=nodes,
        node_dict=node_dict,
        edges=edges,
        uedges=uedges,
        uedge_to_edge=uedge_to_edge,
        demands=demands
    )


>>>>>>> GA_optimization
"""
==================================================
 OUTPUT
==================================================
"""

<<<<<<< HEAD
def write_parameter_results(
    param_name: str,
=======

def write_parameter_results(
    file_name: str,
>>>>>>> GA_optimization
    base_params: GAParams,
    values: List[Union[int, float]],
    results: List[List[float]],
    runtimes: List[List[float]]
):
    """
    Write parameter analysis results to file
    Format:
    Line 1: JSON string of base parameters
    Line 2: JSON list of tested values
    Line 3: JSON list of results lists
    Line 4: JSON list of runtime lists
    """
<<<<<<< HEAD
    filename = f"{param_name}.txt"
    filepath = os.path.join("output", filename)
    
=======
    filename = f"{file_name}.txt"
    filepath = os.path.join("output", filename)

>>>>>>> GA_optimization
    with open(filepath, 'w') as f:
        f.write(json.dumps(base_params.__dict__) + "\n")
        f.write(json.dumps(values) + "\n")
        f.write(json.dumps(results) + "\n")
        f.write(json.dumps(runtimes) + "\n")
<<<<<<< HEAD
    print(f"Saved results for {param_name} to {filepath}")
=======
    print(f"Saved results for {file_name} to {filepath}")

>>>>>>> GA_optimization

def read_parameter_results(filepath: str):
    """
    Read parameter analysis results from file
    Returns tuple: (base_params, values, results, runtimes)
    """
    with open(filepath, 'r') as f:
        # Read all lines
        lines = f.readlines()
        base_params_dict = json.loads(lines[0].strip())
        base_params = GAParams(**base_params_dict)
        values = json.loads(lines[1].strip())
        results = json.loads(lines[2].strip())
        runtimes = json.loads(lines[3].strip())
<<<<<<< HEAD
    
    return base_params, values, results, runtimes
=======

    return base_params, values, results, runtimes
>>>>>>> GA_optimization
