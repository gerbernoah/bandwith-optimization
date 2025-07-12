from collections import defaultdict, deque
import re

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