"""
Data structures for results of optimization algorithms.

types: ResultGA, ResultIP
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class ResultGA:
    total_runtime: float
    total_cost: float

@dataclass
class ResultIP:
    total_runtime: float
    total_cost: float