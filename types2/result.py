"""
Data structures for results of optimization algorithms.

types: ResultGA, ResultIP
"""

from dataclasses import dataclass
from typing import List

@dataclass
class ResultGA:
    total_runtime: float
    total_cost: float
    violations: int # number of edges that have more flow than capacity

@dataclass
class ResultIP:
    total_runtime: float
    total_cost: float