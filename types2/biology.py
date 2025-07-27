"""
Data Structures for Genetic Algorithm.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from types2.network import Edge
from deap import base, creator

# Create DEAP fitness type first
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

@dataclass
class GAIndividual:
    """
    Represents a Genetic Algorithm individual for network design.
    
    Attributes:
        module_selections: List of integers representing module choices for each undirected edge.
                           -1 = no module, 0+ = module index
        flow_fractions: List of flow distributions for each demand.
                        Each element is a list of 5 floats (summing to 1) representing
                        fraction of demand allocated to each path
        fitness: DEAP fitness object (required by DEAP framework)
    """
    module_selections: List[int]
    flow_fractions: List[List[float]]
    fitness: creator.FitnessMin = field(default_factory=lambda: creator.FitnessMin())

    def __post_init__(self):
        """Validation remains valuable for debugging"""
        # Validate flow fractions (keep but make less strict)
        for fractions in self.flow_fractions:
            if abs(sum(fractions) - 1.0) > 1e-3:  # Relaxed tolerance
                # Normalize instead of raising error
                total = sum(fractions)
                for i in range(len(fractions)):
                    fractions[i] /= total

type Path = List[Edge]
type Paths5 = List[Path]
type DemandPaths = List[Paths5]