import pulp
from parsing import parse_nodes, parse_links, parse_demands
import matplotlib.pyplot as plt
import numpy as np

"""
So we currently have some problems, we should redefine the model a little bit differently to handle the given undericted edges.
We should create two edges for each edge, AND CREATE A CONSTRAINT that makes sure that the sum of the flows on both edges is less than the capacity of the edge,
instead of having just one edge I guess.
"""

nodeids = parse_nodes("nodes.txt")
linkids, routing_costs, modules = parse_links("links.txt")
demands = parse_demands("demands.txt")

# Sample data (replace with your actual data)
# V = [1, 2, 3]
# E = {
#     1: (1, 2, 5),  # edge_id: (u, v, routing_cost)
#     2: (2, 3, 4),
#     3: (1, 3, 10)
# }
# M = {
#     1: [(100, 500), (200, 800)],  # edge_id: list of (capacity, module_cost)
#     2: [(100, 300)],
#     3: [(150, 400)]
# }
# D = [
#     (1, 3, 150),  # (source, target, demand)
# ]

V = [i for i in range(len(nodeids))]  # Node IDs from 0 to n-1
E = {
    f"{linkids[link_id]}_fwd": (int(src), int(tgt), routing_costs[linkids[link_id]])
    for link_id in linkids
    for src, tgt in [link_id.split()]
} | {
    f"{linkids[link_id]}_rev": (int(tgt), int(src), routing_costs[linkids[link_id]])
    for link_id in linkids
    for src, tgt in [link_id.split()]
}
M = modules
D = demands

print("Nodes:", V)
print("Edges:", E)
print("Modules:", M)
print("Demands:", D)

# Initialize problem
prob = pulp.LpProblem("NetworkDesign", pulp.LpMinimize)

# Decision variables
x = {}  # flow variables: x[e][s][t]
y = {}  # module installation: y[e][m]

# Create flow variables for each directed edge
for e in E:
    x[e] = {}
    for s, t, d in D:
        x[e][(s, t)] = pulp.LpVariable(f"x_{e}_{s}_{t}", lowBound=0)

# Create module variables for each base (undirected) edge
for base_id in M:
    y[base_id] = {}
    for idx, (cap, cost) in enumerate(M[base_id]):
        y[base_id][idx] = pulp.LpVariable(f"y_{base_id}_{idx}", cat="Binary")

# Objective: minimize routing cost (and optionally module cost)
routing_cost = pulp.lpSum(
    # flow through edge e of flow (s, t) multiplied by routing cost
    x[e][(s, t)] * E[e][2]
    for e in E
    for (s, t, _) in D
)
module_cost = pulp.lpSum(
    y[e][m] * M[e][m][1]  # decision * cost of module indexed "m" on edge e
    for e in M  # iterate over base edge IDs in M
    for m in y[e]
)
prob += routing_cost + module_cost

# Flow conservation
for s, t, d_val in D:
    for v in V:
        inflow = pulp.lpSum(
            x[e][(s, t)] for e in E if E[e][1] == v
        )
        outflow = pulp.lpSum(
            x[e][(s, t)] for e in E if E[e][0] == v
        )

        if v == s:
            prob += (outflow - inflow == d_val), f"flow_src_{s}_{t}_{v}"
        elif v == t:
            prob += (inflow - outflow == d_val), f"flow_tgt_{s}_{t}_{v}"
        else:
            prob += (inflow == outflow), f"flow_bal_{s}_{t}_{v}"

# Capacity constraints
for base_edge_id in M:  # iterate over base edge IDs
    # Sum flows on both directions of the undirected edge
    total_flow_on_edge = pulp.lpSum(
        x[f"{base_edge_id}_fwd"][(s, t)] + x[f"{base_edge_id}_rev"][(s, t)]
        for (s, t, _) in D
    )
    prob += (
        total_flow_on_edge <=  # the total flow on both directions of the undirected edge
        # RHS: sum of selected module capacities on edge
        pulp.lpSum(y[base_edge_id][m] * M[base_edge_id][m][0]
                   for m in y[base_edge_id])
    ), f"cap_{base_edge_id}"

# One module per edge max
for base_edge_id in M:  # iterate over base edge IDs
    prob += (
        pulp.lpSum(y[base_edge_id][m] for m in y[base_edge_id]) <= 1
    ), f"one_module_{base_edge_id}"

# Solve
prob.solve()

# Output
print("Status:", pulp.LpStatus[prob.status])
print("Total cost:", pulp.value(prob.objective))

for base_edge_id in M:  # iterate over base edge IDs
    for m in y[base_edge_id]:
        if pulp.value(y[base_edge_id][m]) > 0.5:
            print(
                f"Install module {m} on edge {base_edge_id} with cap {M[base_edge_id][m][0]}")

for e in E:
    for (s, t, _) in D:
        val = pulp.value(x[e][(s, t)])
        if val > 0:
            print(f"Flow {val} from {s}->{t} via edge {e}")
