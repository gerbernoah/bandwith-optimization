"""
Data Structures for Genetic Algorithm.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from types2.network import Edge
from deap import base, creator
from enum import Enum

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
    fitness: creator.FitnessMin = field(default_factory=lambda: creator.FitnessMin()) # type: ignore

    def __post_init__(self):
        """Validation remains valuable for debugging"""
        # Validate flow fractions (keep but make less strict)
        for fractions in self.flow_fractions:
            if abs(sum(fractions) - 1.0) > 1e-3:  # Relaxed tolerance
                # Normalize instead of raising error
                total = sum(fractions)
                for i in range(len(fractions)):
                    fractions[i] /= total

# types for the 5 shortest demand paths
type Path = List[Edge]
type Paths5 = List[Path]
type DemandPaths = List[Paths5]

# GA create_individual strategies
class InitStrategy(Enum):
    RANDOM = 1      # randomly create individuals
    MIN_MODULE = 2  # take lowest capacity module
    MAX_MODULE = 3  # take highest capacity module


@dataclass
class GAParams:
    """
    - npop: initial population size
    - cxpb: cross over probability
    - mutpb: mutation probability
    - ngen: number of generations
    - penalty_coeff: penalty for each flow unit over the capacity of an edge
    - max_module_ratio: initial population % with maximum capacity modules selected
    - min_module_ratio: initial population % with minimal capacity modules selected
    - muLambda = (mu, lambda): mu*npop = nr of individuals selected per gen., lambda*npop = number of offspring created each gen.
<<<<<<< HEAD
=======
    - tournsize: fkjdslaöjfdklösa
    - indp: independent probability, the probability applied to each gene independently during mutation
>>>>>>> GA_optimization
    """
    npop: int
    cxpb: float
    mutpb: float
    ngen: int
    penalty_coeff: float
    max_module_ratio: float
    min_module_ratio: float
    muLambda: Tuple[float, float]
<<<<<<< HEAD
=======
    tournsize: int
    indp: float
>>>>>>> GA_optimization

    def toString(self):
        return f"GAParams: {self.npop}, {self.cxpb}, {self.mutpb}, {self.ngen}, {self.penalty_coeff}, {self.max_module_ratio}, {self.min_module_ratio}, {self.muLambda}"