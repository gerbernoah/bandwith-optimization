"""
Data structures for representing a network.

types: Node, NodeDict, Demand, Link, Module, Network
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import re
import os

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
    demand_value: float

    # unique ID counter
    _id_counter: int = 0

    # assign unique ID to each node after Node is created
    def __post_init__(self):
        self.id = Demand._id_counter
        Demand._id_counter += 1

@dataclass
class Module:
    """Represents a link module. Each link can have multiple module options, but at most one can be installed."""
    capacity: float
    cost: float

@dataclass
class Link:
    id: int = field(init=False) # special field not passed to constructor
    source: Node
    target: Node
    routing_cost: float
    module_options: List[Module]

    # unique ID counter
    _id_counter: int = 0

    # assign unique ID to each node after Node is created
    def __post_init__(self):
        self.id = Link._id_counter
        Link._id_counter += 1

@dataclass
class Network:
    links: List[Link]
    nodes: List[Node]
    demands: List[Demand]
    node_dict: NodeDict