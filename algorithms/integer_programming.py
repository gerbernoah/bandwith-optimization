import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple

# Data Structures & Utilities
from types2.network import Node, Demand, Edge, Module, NodeDict, UEdge, UEdgeToEdge, Network
from types2.result import ResultIP
from utilities.printing import print_title

def create_model(network: Network) -> gp.Model:
    nodes, node_dict, edges, uedges, uedge_to_edge, demands = network.unpack()

    """
    ==================================================
        MODEL INITIALIZATION
    ==================================================
    """

    # create gurubipy model
    model = gp.Model("NetworkDesign")

    """
    ==================================================
        VARIABLE CREATION
    ==================================================
    """

    # Flow Variables: x[d, e]
    # where d = demand ID, e = directed edge ID
    x = {}
    for demand in demands:
        for edge in edges:
            x[demand.id, edge.id] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"x_{demand.id}_{edge.id}")

    # Module installation Variables: y[ue, i]
    # where ue = undirected edge ID, i = module index in the Module list of the edge
    y = {}
    for uedge in uedges:
        for module in uedge.module_options:
            y[uedge.id, module.index] = model.addVar(vtype=GRB.BINARY, name=f"y_{uedge.id}_{module.index}")

    """
    ==================================================
        OBJECTIVE FUNCTION
    ==================================================
    """
    routing_cost = gp.quicksum(x[d.id, e.id] * e.uEdge.routing_cost for d in demands for e in edges)
    module_cost = gp.quicksum(y[ue.id, m.index] * m.cost for ue in uedges for m in ue.module_options)
    model.setObjective(routing_cost + module_cost, GRB.MINIMIZE)

    """
    ==================================================
        CONSTRAINTS
    ==================================================
    """

    # Flow Conservation (on each node) per Demand
    for demand in demands:
        for node in nodes:
            inflow = gp.quicksum(x[demand.id, e.id] for e in edges if e.target.id == node.id)
            outflow = gp.quicksum(x[demand.id, e.id] for e in edges if e.source.id == node.id)

            if node.id == demand.source.id:
                model.addConstr(outflow - inflow == demand.value, name=f"flow_src_{demand.id}_{node.id}")
            elif node.id == demand.target.id:
                model.addConstr(inflow - outflow == demand.value, name=f"flow_tgt_{demand.id}_{node.id}")
            else:
                model.addConstr(inflow == outflow, name=f"flow_bal_{demand.id}_{node.id}")

    # Capacity Constraint per Undirected Edge
    for uedge in uedges:
        e1, e2 = uedge_to_edge[uedge.id]
        flow = gp.quicksum(x[d.id, e1.id] + x[d.id, e2.id] for d in demands)
        capacity = gp.quicksum(y[uedge.id, m.index] * m.capacity for m in uedge.module_options)

        model.addConstr(flow <= capacity, name=f"cap_{uedge.id}")

    # One Module Constraint per Undirected Edge
    for uedge in uedges:
        module_count = gp.quicksum(y[uedge.id, m.index] for m in uedge.module_options)

        model.addConstr(module_count <= 1, name=f"one_module_{uedge.id}")

    return model

"""
==================================================
    RUN OPTIMIZATION
==================================================
"""

def run_IP(
        network: Network,
        log = True,
        timeLimit = 5*60,
        gapLimit = 0.1
    ):
    """
    Args:
    - log: whether Gurubi should print information while processing
    - timeLimit: timeLimit for the Integer Program
    - gapLimit: gap limit for the Integer Program
    """
    print_title("INTEGER PROGRAMMING")
    print("=== LOGGING ON ===" if log else "=== LOGGING OFF ===")

    model = create_model(network)
    model.setParam('OutputFlag', (1 if log else 0))
    model.setParam("TimeLimit", timeLimit)
    model.setParam("MIPGap", gapLimit)
    model.optimize()

    status = model.Status
    total_cost = float('inf')

    if status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        total_cost = model.ObjVal

    total_runtime = model.Runtime

    return ResultIP(
        total_runtime=total_runtime,
        total_cost=total_cost
    )