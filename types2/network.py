"""
Data structures for representing a network.

types: Node, NodeDict, Demand, Edge, Module, Network
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class Node:
    id: int = field(init=False) # special field not passed to constructor
    name: str

    # unique ID counter
    _id_counter: int = 0

    # assign unique ID to each node after Node is created
    def __post_init__(self):
        self.id = Node._id_counter
        Node._id_counter += 1

# mapping from node name to Noede object
type NodeDict = Dict[str, Node]

@dataclass
class Demand:
    id: int = field(init=False) # special field not passed to constructor
    source: Node
    target: Node
    value: float

    # unique ID counter
    _id_counter: int = 0

    # assign unique ID to each node after Demans is created
    def __post_init__(self):
        self.id = Demand._id_counter
        Demand._id_counter += 1

@dataclass
class Module:
    """Represents a module. Each edge can have multiple module options, but at most one can be installed."""
    capacity: float
    cost: float
    index: int # index of this module in the module_options list of the Edge

@dataclass
class UEdge:
    """Represents an undirected edge in the network. Used by (directed) Edge objects."""
    id: int = field(init=False) # special field not passed to constructor
    routing_cost: float
    module_options: List[Module]

    # unique ID counter
    _id_counter: int = 0

    # assign unique ID to each node after Edge is created
    def __post_init__(self):
        self.id = Edge._id_counter
        Edge._id_counter += 1

@dataclass
class Edge:
    """
    Represents a directed edge in the network.

    Usage:
    - for undirected edge, create two Edge objects asynchronously
    - that way, for each uneven ID i, edges with IDs i and i+1 are the same undirected edge
    """
    id: int = field(init=False) # special field not passed to constructor
    source: Node
    target: Node
    uEdge: UEdge

    # unique ID counter
    _id_counter: int = 0

    # assign unique ID to each node after Edge is created
    def __post_init__(self):
        self.id = Edge._id_counter
        Edge._id_counter += 1

type UEdgeToEdge = Dict[int, Tuple[Edge, Edge]]

@dataclass
class Network:
    nodes: List[Node]
    node_dict: NodeDict
    edges: List[Edge]
    uedges: List[UEdge]
    uedge_to_edge: UEdgeToEdge
    demands: List[Demand]

    def unpack(self):
        return (self.nodes, self.node_dict, self.edges,
                    self.uedges, self.uedge_to_edge, self.demands)