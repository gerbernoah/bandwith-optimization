import pulp
from collections import defaultdict, deque
import re
import os

# --- Parsing Functions ---

# Note that some demands are "skipped" by our given input, i.e. D166 is not the 166th demand, could be the 151st demand or similar.

def parse_nodes(filename):
    nodes = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # Match: <node_id> ( <longitude> <latitude> )
            m = re.match(r"(\w+)\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", line)
            if not m:
                continue
            name, lon, lat = m.groups()
            nodes[name] = (float(lon), float(lat))
    return nodes

def parse_links(filename):
    links = {}
    modules = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # Match: <link_id> ( <source> <target> ) <pre_inst_cap> <pre_inst_cap_cost> <routing_cost> <setup_cost> ( {<mod_cap> <mod_cost>}* )
            m = re.match(
                r"\w+\s*\(\s*(\w+)\s+(\w+)\s*\)\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\((.*)\)",
                line
            )
            if not m:
                continue
            src, tgt, pre_cap, pre_cap_cost, routing_cost, setup_cost, mods = m.groups()
            link_key = (src, tgt)
            links[link_key] = {
                'routing_cost': float(routing_cost),
                'setup_cost': float(setup_cost),
                'pre_cap': float(pre_cap),
                'pre_cap_cost': float(pre_cap_cost)
            }
            # Parse modules: <mod_cap> <mod_cost> pairs
            mod_list = []
            for mod in re.findall(r"([-\d.]+)\s+([-\d.]+)", mods):
                cap, cost = map(float, mod)
                mod_list.append((cap, cost))
            modules[link_key] = mod_list
    return links, modules

def parse_demands(filename):
    demands = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("<"):
                continue
            # Match: <demand_id> ( <source> <target> ) <routing_unit> <demand_value> <max_path_length>
            m = re.match(
                r"\w+\s*\(\s*(\w+)\s+(\w+)\s*\)\s*\d+\s+([-\d.]+)",
                line
            )
            if not m:
                continue
            src, tgt, val = m.groups()
            demands.append({'src': src, 'tgt': tgt, 'val': float(val)})
    return demands


def check_capacities(links, modules, demands):
    """Check if total installable capacity on any link is enough for all flows."""
    # Compute max possible capacity for each link
    max_caps = {}
    for (i, j) in links:
        pre = links[(i, j)]['pre_cap']
        mod_caps = sum(cap for cap, cost in modules[(i, j)])
        max_caps[(i, j)] = pre + mod_caps
    # For each demand, check if there is at least one path with enough capacity
    # (This is a simple check: sum of all demands <= sum of all max_caps)
    total_demand = sum(d['val'] for d in demands)
    total_capacity = sum(max_caps.values())
    if total_demand > total_capacity:
        print(f"WARNING: Total demand ({total_demand}) exceeds total installable capacity ({total_capacity})")
    else:
        print("Total installable capacity is sufficient for total demand.")

def find_connected_components(nodes, links):
    """Find and print all connected components (groups of nodes) in the directed graph."""
    visited = set()
    components = []

    # Build adjacency list (undirected for weak connectivity)
    adj = {n: set() for n in nodes}
    for (src, tgt) in links:
        adj[src].add(tgt)
        adj[tgt].add(src)

    for node in nodes:
        if node not in visited:
            group = set()
            queue = deque([node])
            visited.add(node)
            while queue:
                n = queue.popleft()
                group.add(n)
                for neighbor in adj[n]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(group)

    print(f"Number of connected components: {len(components)}")
    for idx, group in enumerate(components, 1):
        print(f"Component {idx} ({len(group)} nodes): {sorted(group)}")

# --- Main Model ---

def main():
    nodes = parse_nodes('nodes.txt')
    links, modules = parse_links('links.txt')
    demands = parse_demands('demands.txt')

    print(f"Loaded {len(nodes)} nodes, {len(links)} links, {len(demands)} demands")

    # Connectivity and capacity checks
    check_capacities(links, modules, demands)
    find_connected_components(nodes, links)

    # Build sets
    N = set(nodes)
    L = set(links)
    D = list(range(len(demands)))

    # Variables
    prob = pulp.LpProblem("NetworkDesign", pulp.LpMinimize)

    # x[(i,j),d]: flow of demand d on link (i,j)
    x = pulp.LpVariable.dicts("x", ((i, j, d) for (i, j) in L for d in D), lowBound=0, cat='Continuous')

    # y[(i,j),m]: number of modules m installed on link (i,j)
    y = {}
    for (i, j) in L:
        for m, (cap, cost) in enumerate(modules[(i, j)]):
            y[(i, j, m)] = pulp.LpVariable(f"y_{i}_{j}_mod{m}", lowBound=0, cat='Integer')

    # z[(i,j)]: 1 if any module is installed on link (i,j)
    z = pulp.LpVariable.dicts("z", L, cat='Binary')

    # --- Objective ---
    routing_cost = pulp.lpSum(
        x[(i, j, d)] * links[(i, j)]['routing_cost']
        for (i, j) in L for d in D
    )
    setup_cost = pulp.lpSum(
        z[(i, j)] * links[(i, j)]['setup_cost']
        for (i, j) in L
    )
    module_cost = pulp.lpSum(
        y[(i, j, m)] * modules[(i, j)][m][1]
        for (i, j) in L for m in range(len(modules[(i, j)]))
    )
    # Optionally add pre-installed capacity cost
    pre_cap_cost = pulp.lpSum(
        links[(i, j)]['pre_cap_cost'] for (i, j) in L if links[(i, j)]['pre_cap'] > 0
    )
    prob += routing_cost + setup_cost + module_cost + pre_cap_cost

    # --- Constraints ---

    # Flow conservation
    for d, demand in enumerate(demands):
        for n in N:
            inflow = pulp.lpSum(x[(i, j, d)] for (i, j) in L if j == n)
            outflow = pulp.lpSum(x[(i, j, d)] for (i, j) in L if i == n)
            if n == demand['src']:
                prob += (outflow - inflow == demand['val'])
            elif n == demand['tgt']:
                prob += (inflow - outflow == demand['val'])
            else:
                prob += (outflow - inflow == 0)

    # Capacity constraints (include pre-installed capacity)
    for (i, j) in L:
        total_cap = links[(i, j)]['pre_cap'] + pulp.lpSum(
            y[(i, j, m)] * modules[(i, j)][m][0]
            for m in range(len(modules[(i, j)]))
        )
        total_flow = pulp.lpSum(x[(i, j, d)] for d in D)
        prob += total_flow <= total_cap

        # Setup cost only if any module is installed
        prob += total_cap - links[(i, j)]['pre_cap'] <= 1e6 * z[(i, j)]

    # --- Solve ---
    prob.solve()

    # --- Output ---
    output_lines = []
    status = pulp.LpStatus[prob.status]
    total_cost = pulp.value(prob.objective) if status == "Optimal" else "N/A"
    output_lines.append(f"Status: {status}")
    output_lines.append(f"Total cost: {total_cost}\n")

    output_lines.append("Used links and installed capacities:")
    if status == "Optimal":
        for (i, j) in L:
            cap = links[(i, j)]['pre_cap']
            for m in range(len(modules[(i, j)])):
                val = pulp.value(y[(i, j, m)])
                if val is None:
                    val = 0.0
                cap += val * modules[(i, j)][m][0]
            if cap > 1e-3:
                output_lines.append(f"{i} -> {j}: {cap}")
    else:
        output_lines.append("No solution found.")

    output_lines.append("\nPaths for each demand:")
    if status == "Optimal":
        for d, demand in enumerate(demands):
            output_lines.append(f"Demand {demand['src']} -> {demand['tgt']} ({demand['val']}):")
            for (i, j) in L:
                flow = pulp.value(x[(i, j, d)])
                if flow is not None and flow > 1e-3:
                    output_lines.append(f"  {i} -> {j}: {flow}")
    else:
        output_lines.append("No paths found.")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/integer-programming.txt", "w") as f:
        f.write("\n".join(output_lines))

    # Also print status and total cost to console
    print(f"Status: {status}")
    print(f"Total cost: {total_cost}")
    print('Output written to outputs/integer-programming.txt')

if __name__ == "__main__":
    main()