import gurobipy as gp
from gurobipy import GRB
import importlib
import numpy as np
import sys
import os

# this import works when this code is run from root directory
from types.result import ResultIP
from types.network import NetworkGraph


def run_integer_programming(nodes_file="nodes.txt", links_file="links.txt", demands_file="demands.txt", time_limit=None, mip_gap=0.3, msg=True):
    """
    Run integer programming optimization using Gurobi (no PuLP).
    Args:
        nodes_file: Path to nodes file
        links_file: Path to links file
        demands_file: Path to demands file
        time_limit: Maximum time in seconds (None for no limit)
        mip_gap: Relative MIP gap (default 0.3 = 30%)
        msg: Whether to show solver messages
    Returns:
        dict: Contains optimization results
    """
    # Load network
    network = NetworkGraph()
    network.load_from_files(nodes_file, links_file, demands_file)

    nodeids = {node_id: i for i, node_id in enumerate(network.nodes.keys())}
    V = list(range(len(network.nodes)))
    link_ids = list(network.links.keys())
    # Directed edges
    E = {}
    for link_id, link in network.links.items():
        src_idx = nodeids[link.source]
        tgt_idx = nodeids[link.target]
        E[f"{link_id}_fwd"] = (src_idx, tgt_idx, link.routing_cost)
        E[f"{link_id}_rev"] = (tgt_idx, src_idx, link.routing_cost)
    # Modules
    M = {link_id: link.get_total_capacity_options() for link_id, link in network.links.items()}
    # Demands
    D = []
    for demand_id, demand in network.demands.items():
        src_idx = nodeids[demand.source]
        tgt_idx = nodeids[demand.target]
        D.append((src_idx, tgt_idx, demand.demand_value))

    model = gp.Model("NetworkDesign")
    if not msg:
        model.setParam('OutputFlag', 0)
    if time_limit:
        model.setParam('TimeLimit', time_limit)
    if mip_gap:
        model.setParam('MIPGap', mip_gap)

    # Variables
    x = {}  # flow variables: x[e,s,t]
    for e in E:
        for s, t, dval in D:
            x[e, s, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"x_{e}_{s}_{t}")
    y = {}  # module install: y[edge, m]
    for link_id in M:
        for m, (cap, cost) in enumerate(M[link_id]):
            y[link_id, m] = model.addVar(vtype=GRB.BINARY, name=f"y_{link_id}_{m}")

    model.update()

    # Objective: routing cost + module cost
    routing_cost = gp.quicksum(x[e, s, t] * E[e][2] for e in E for s, t, _ in D)
    module_cost = gp.quicksum(y[link_id, m] * M[link_id][m][1] for link_id in M for m in range(len(M[link_id])))
    model.setObjective(routing_cost + module_cost, GRB.MINIMIZE)

    # Flow conservation
    for s, t, dval in D:
        for v in V:
            inflow = gp.quicksum(x[e, s, t] for e in E if E[e][1] == v)
            outflow = gp.quicksum(x[e, s, t] for e in E if E[e][0] == v)
            if v == s:
                model.addConstr(outflow - inflow == dval, name=f"flow_src_{s}_{t}_{v}")
            elif v == t:
                model.addConstr(inflow - outflow == dval, name=f"flow_tgt_{s}_{t}_{v}")
            else:
                model.addConstr(inflow == outflow, name=f"flow_bal_{s}_{t}_{v}")

    # Capacity constraints (sum of flows in both directions <= installed capacity)
    for link_id in M:
        total_flow = gp.quicksum(x[f"{link_id}_fwd", s, t] + x[f"{link_id}_rev", s, t] for s, t, _ in D)
        total_cap = gp.quicksum(y[link_id, m] * M[link_id][m][0] for m in range(len(M[link_id])))
        model.addConstr(total_flow <= total_cap, name=f"cap_{link_id}")

    # One module per edge
    for link_id in M:
        model.addConstr(gp.quicksum(y[link_id, m] for m in range(len(M[link_id]))) == 1, name=f"one_module_{link_id}")

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


def main():
    print("Running Integer Programming Optimization with Gurobi...")
    results = run_integer_programming(
        nodes_file="../nodes.txt",
        links_file="../links.txt",
        demands_file="../demands.txt",
        time_limit=20,  # seconds
        mip_gap=0.3,
        msg=True
    )
    print(f"\nOptimization Results:")
    print(f"Status: {results['status']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Best objective: {results['best_fitness']}")
    print(f"Best solution: {results['best_solution']}")
    return results


if __name__ == "__main__":
    main()
