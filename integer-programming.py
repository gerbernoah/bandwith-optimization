import pulp
from parsing import parse_nodes, parse_links, parse_demands

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
E = {linkids[k]: (k.split()[0], k.split()[1], routing_costs[linkids[k]]) for k in linkids.keys()}  # Edge ID to (source, target, routing cost)
M = modules
D = demands

# Initialize problem
prob = pulp.LpProblem("NetworkDesign", pulp.LpMinimize)

# Decision variables
x = {}  # flow variables: x[e][s][t]
y = {}  # module installation: y[e][m]

for e in E:
    x[e] = {}
    for s, t, d in D:
        x[e][(s, t)] = pulp.LpVariable(f"x_{e}_{s}_{t}", lowBound=0)
    y[e] = {}
    for idx, (cap, cost) in enumerate(M[e]):
        y[e][idx] = pulp.LpVariable(f"y_{e}_{idx}", cat="Binary")

# Objective: minimize routing cost (and optionally module cost)
routing_cost = pulp.lpSum(
    x[e][(s, t)] * E[e][2]
    for e in E
    for (s, t, _) in D
)
module_cost = pulp.lpSum(
    y[e][m] * M[e][m][1]
    for e in E
    for m in y[e]
)
prob += routing_cost + module_cost

# Flow conservation
for s, t, d_val in D:
    for v in V:
        inflow = pulp.lpSum(
            x[e][(s, t)] for e in E if E[e][1] == v or E[e][0] == v
        )
        outflow = pulp.lpSum(
            x[e][(s, t)] for e in E if E[e][0] == v or E[e][1] == v
        )

        if v == s:
            prob += (outflow - inflow == d_val), f"flow_src_{s}_{t}_{v}"
        elif v == t:
            prob += (inflow - outflow == d_val), f"flow_tgt_{s}_{t}_{v}"
        else:
            prob += (inflow == outflow), f"flow_bal_{s}_{t}_{v}"

# Capacity constraints
for e in E:
    prob += (
        pulp.lpSum(x[e][(s, t)] for (s, t, _) in D) <=
        pulp.lpSum(y[e][m] * M[e][m][0] for m in y[e])
    ), f"cap_{e}"

# One module per edge max
for e in E:
    prob += (
        pulp.lpSum(y[e][m] for m in y[e]) <= 1
    ), f"one_module_{e}"

# Solve
prob.solve()

# Output
print("Status:", pulp.LpStatus[prob.status])
print("Total cost:", pulp.value(prob.objective))

for e in E:
    for m in y[e]:
        if pulp.value(y[e][m]) > 0.5:
            print(f"Install module {m} on edge {e} with cap {M[e][m][0]}")

for e in E:
    for (s, t, _) in D:
        val = pulp.value(x[e][(s, t)])
        if val > 0:
            print(f"Flow {val} from {s}->{t} via edge {e}")
