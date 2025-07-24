from collections import defaultdict, deque
import re

# --- Parsing Functions ---

# Note that some demands are "skipped" by our given input, i.e. D166 is not the 166th demand, could be the 151st demand or similar.

def parse_nodes(filename):
    # returns dictionary that maps node name to id
    nodes = {}
    nodeids: dict[str, int] = {}
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
            nodeids[name] = len(nodeids)  # Assign a unique ID starting from 0
            nodes[name] = (float(lon), float(lat))
    return nodeids

def parse_links(filename):
    # returns three dictionaries: one that maps two vertices to a linkid, one that maps linkid to a routing cost, one that maps link id to a list of modules (capacity, cost)
    nodeids = parse_nodes("nodes.txt")  # Assuming nodes are in a file named "nodes.txt"
    modules: dict[int, list[tuple[float, float]]] = {}
    linkids: dict[str, int] = {}
    routing_costs: dict[int, int] = {}
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
            linkid = len(linkids)  # Assign a unique ID starting from 0
            src = nodeids[src]
            tgt = nodeids[tgt]
            linkids[f"{src} {tgt}"] = linkid
            linkids[f"{tgt} {src}"] = linkid # Add reverse link as well
            routing_costs[linkid] = int(float(routing_cost))
            # Parse modules: <mod_cap> <mod_cost> pairs
            mod_list = []
            for mod in re.findall(r"([-\d.]+)\s+([-\d.]+)", mods):
                cap, cost = map(float, mod)
                mod_list.append((cap, cost))
            modules[linkid] = mod_list
    return linkids, routing_costs, modules

def parse_demands(filename):
    demands = []
    nodeids = parse_nodes("nodes.txt")  # Assuming nodes are in a file named "nodes.txt"
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
            demands.append((nodeids[src], nodeids[tgt], int(float(val))))
    return demands