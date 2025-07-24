import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import pulp
from parsing import parse_nodes, parse_links, parse_demands


def plot_network_results():
    """Plot the network optimization results showing flows and installed modules."""

    # Parse data (same as in integer-programming.py)
    nodeids = parse_nodes("nodes.txt")
    linkids, routing_costs, modules = parse_links("links.txt")
    demands = parse_demands("demands.txt")

    V = [i for i in range(len(nodeids))]
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

    # Solve the optimization problem (same as in integer-programming.py)
    prob = pulp.LpProblem("NetworkDesign", pulp.LpMinimize)

    x = {}
    y = {}

    # Create flow variables for each directed edge
    for e in E:
        x[e] = {}
        for s, t, d in D:
            x[e][(s, t)] = pulp.LpVariable(f"x_{e}_{s}_{t}", lowBound=0)

    # Create module variables for each base (undirected) edge
    for base_id in M:
        y[base_id] = {}
        for idx, (cap, cost) in enumerate(M[base_id]):
            y[base_id][idx] = pulp.LpVariable(
                f"y_{base_id}_{idx}", cat="Binary")

    # Objective
    routing_cost = pulp.lpSum(
        x[e][(s, t)] * E[e][2]
        for e in E
        for (s, t, _) in D
    )
    module_cost = pulp.lpSum(
        y[e][m] * M[e][m][1]
        for e in M
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
    for base_edge_id in M:
        total_flow_on_edge = pulp.lpSum(
            x[f"{base_edge_id}_fwd"][(s, t)] + x[f"{base_edge_id}_rev"][(s, t)]
            for (s, t, _) in D
        )
        prob += (
            total_flow_on_edge <=
            pulp.lpSum(y[base_edge_id][m] * M[base_edge_id][m][0]
                       for m in y[base_edge_id])
        ), f"cap_{base_edge_id}"

    # One module per edge max
    for base_edge_id in M:
        prob += (
            pulp.lpSum(y[base_edge_id][m] for m in y[base_edge_id]) <= 1
        ), f"one_module_{base_edge_id}"

    # Solve
    prob.solve()

    if prob.status != pulp.LpStatusOptimal:
        print(f"Problem status: {pulp.LpStatus[prob.status]}")
        return

    # Collect results
    edge_flows = {}  # base_edge_id -> total_flow
    edge_modules = {}  # base_edge_id -> (module_idx, capacity, cost)

    # Get installed modules
    for base_edge_id in M:
        for m in y[base_edge_id]:
            if pulp.value(y[base_edge_id][m]) > 0.5:
                capacity, cost = M[base_edge_id][m]
                edge_modules[base_edge_id] = (m, capacity, cost)

    # Calculate total flows on each undirected edge
    for base_edge_id in M:
        total_flow = 0
        for (s, t, _) in D:
            fwd_flow = pulp.value(x[f"{base_edge_id}_fwd"][(s, t)]) or 0
            rev_flow = pulp.value(x[f"{base_edge_id}_rev"][(s, t)]) or 0
            total_flow += fwd_flow + rev_flow
        edge_flows[base_edge_id] = total_flow

    # Create simple node positions in a circle
    num_nodes = len(V)
    node_positions = {}
    for i, node in enumerate(V):
        angle = 2 * np.pi * i / num_nodes
        x = np.cos(angle)
        y = np.sin(angle)
        node_positions[node] = (x, y)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Network graph
    ax1.set_aspect('equal')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_title("Network Topology with Flows",
                  fontsize=14, fontweight='bold')

    # Draw nodes
    for node in V:
        x, y = node_positions[node]

        # Color nodes based on demand role
        is_source = any(node == s for s, t, _ in D)
        is_target = any(node == t for s, t, _ in D)

        if is_source and is_target:
            color = 'orange'  # Both source and target
        elif is_source:
            color = 'lightgreen'  # Source
        elif is_target:
            color = 'lightcoral'  # Target
        else:
            color = 'lightblue'  # Regular node

        circle = plt.Circle((x, y), 0.1, color=color, alpha=0.8, zorder=3)
        ax1.add_patch(circle)
        ax1.text(x, y, str(node), ha='center',
                 va='center', fontweight='bold', zorder=4)

    # Draw edges
    for base_edge_id in M:
        edge_parts = base_edge_id.split()
        if len(edge_parts) == 2:
            src, tgt = int(edge_parts[0]), int(edge_parts[1])
            x1, y1 = node_positions[src]
            x2, y2 = node_positions[tgt]

            flow = edge_flows.get(base_edge_id, 0)

            # Line width based on flow
            width = max(1, min(5, flow / 10))

            # Line color based on whether module is installed
            if base_edge_id in edge_modules:
                color = 'red'  # Module installed
                alpha = 0.8
            else:
                color = 'gray'  # No module
                alpha = 0.5

            # Draw the edge
            ax1.plot([x1, x2], [y1, y2], color=color,
                     linewidth=width, alpha=alpha, zorder=1)

            # Add edge label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

            label_parts = []
            if flow > 0:
                label_parts.append(f"Flow: {flow:.1f}")

            if base_edge_id in edge_modules:
                module_idx, capacity, cost = edge_modules[base_edge_id]
                label_parts.append(f"Mod{module_idx}")
                label_parts.append(f"Cap:{capacity}")

            if label_parts:
                ax1.text(mid_x, mid_y, '\n'.join(label_parts),
                         ha='center', va='center', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor="white", alpha=0.8),
                         zorder=2)

    # Add legend for network plot
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                   markersize=10, label='Demand Source'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                   markersize=10, label='Demand Target'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=10, label='Source & Target'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markersize=10, label='Regular Node'),
        plt.Line2D([0], [0], color='red', linewidth=3,
                   label='Edge with Module'),
        plt.Line2D([0], [0], color='gray', linewidth=1,
                   label='Edge without Module')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    ax1.axis('off')

    # Right plot: Summary and bar charts
    ax2.axis('off')

    # Calculate costs
    total_cost = pulp.value(prob.objective)
    total_routing_cost = pulp.value(routing_cost)
    total_module_cost = pulp.value(module_cost)

    # Create summary text
    summary_text = f"""OPTIMIZATION RESULTS
    
Status: {pulp.LpStatus[prob.status]}
Total Cost: ${total_cost:.2f}
Routing Cost: ${total_routing_cost:.2f}
Module Cost: ${total_module_cost:.2f}

DEMANDS:"""

    for i, (s, t, demand) in enumerate(D):
        summary_text += f"\n  {s} → {t}: {demand} units"

    summary_text += "\n\nINSTALLED MODULES:"
    for base_edge_id in M:
        for m in y[base_edge_id]:
            if pulp.value(y[base_edge_id][m]) > 0.5:
                cap, cost = M[base_edge_id][m]
                summary_text += f"\n  Edge {base_edge_id}: Module {m}"
                summary_text += f"\n    Capacity: {cap}, Cost: ${cost}"

    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plt.savefig('network_optimization_results.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed results to console
    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"Routing Cost: ${total_routing_cost:.2f}")
    print(f"Module Cost: ${total_module_cost:.2f}")

    print("\nInstalled Modules:")
    for base_edge_id in M:
        for m in y[base_edge_id]:
            if pulp.value(y[base_edge_id][m]) > 0.5:
                cap, cost = M[base_edge_id][m]
                print(
                    f"  Edge {base_edge_id}: Module {m} (Cap: {cap}, Cost: ${cost})")

    print("\nFlow Details:")
    for e in E:
        for (s, t, _) in D:
            val = pulp.value(x[e][(s, t)])
            if val > 0:
                print(f"  Flow {val:.2f} from {s}→{t} via edge {e}")

    print("\nEdge Utilization:")
    for base_edge_id in M:
        total_flow = edge_flows.get(base_edge_id, 0)
        if base_edge_id in edge_modules:
            _, capacity, _ = edge_modules[base_edge_id]
            utilization = (total_flow / capacity * 100) if capacity > 0 else 0
            print(
                f"  Edge {base_edge_id}: {total_flow:.1f}/{capacity} ({utilization:.1f}% utilized)")
        elif total_flow > 0:
            print(
                f"  Edge {base_edge_id}: {total_flow:.1f} flow (no module installed)")


if __name__ == "__main__":
    plot_network_results()
