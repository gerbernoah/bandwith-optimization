You want to minimize the **total routing cost** while **selecting link capacities (modules)** to handle all demands in a graph. This is a structured **Mixed-Integer Linear Programming (MILP)** problem.

---

## 🧩 **Problem Description**

You have:

* A graph with vertices and edges.
* Each edge has a **routing cost** (per unit of flow).
* You have **capacity modules** you can install on edges, each with a capacity and setup cost.
* You have **demands** between pairs of nodes, with specific traffic requirements.

Your goal is:

> To **decide which modules to install on which edges**, and **how to route the traffic**, so that:
>
> * All demands are satisfied (i.e., routed from source to target).
> * Flows don't exceed installed capacity.
> * **Total routing cost is minimized.** (You can also include module costs if desired.)

---

## 🔢 Sets and Parameters

Let’s define your input sets:

| Set | Description                                                                                                                                                      |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $V$ | Set of vertices (e.g., $\{1, 2, 3, 4\}$)                                                                                                                         |
| $E$ | Set of edges. Each element is $(e, \{i,j\}, c_e)$, where $e$ is edge ID, $\{i,j\} \subset V$, and $c_e \in \mathbb{N}$ is the routing cost per unit on edge $e$. |
| $M$ | Set of capacity modules. Each element is $(e, u, m)$: module for edge $e$, with capacity $u$, and setup cost $m$.                                                |
| $D$ | Set of demands. Each element is $(s, t, d_{st})$: source node, target node, and demand amount.                                                                   |

---

## ✅ Variables

We define two types of variables:

1. **Flow variables** $x_e^{st} \geq 0$:

   * How much flow of demand from $s$ to $t$ goes through edge $e$.

2. **Module selection variables** $y_{e,m} \in \{0,1\}$:

   * Whether module $m$ (associated with edge $e$) is installed.

---

## 🎯 Objective Function

Minimize the **total routing cost**:

$$
\min (( \sum_{(s,t,d_{st}) \in D} \sum_{e \in E} c_e \cdot x_e^{st}) + \sum_{(e,u,m) \in M} m \cdot y_{e,m})
$$

> Each unit of demand routed through an edge incurs a cost $c_e$, summed over all edges and all demands, adding module setup cost to the equation.

---

## 📏 Constraints

### 1. **Flow Conservation**

For each demand $(s, t, d_{st})$, we ensure the flow behaves like a proper commodity:

For every node $v \in V$:

$$
\sum_{e \text{ into } v} x_e^{st} - \sum_{e \text{ out of } v} x_e^{st} =
\begin{cases}
-d_{st} & \text{if } v = s \ (\text{source}) \\
+d_{st} & \text{if } v = t \ (\text{target}) \\
0 & \text{otherwise}
\end{cases}
$$

This ensures:

* Flow starts at the source.
* Flow ends at the target.
* No creation or destruction of flow elsewhere.

If your graph is undirected, you treat each edge as bidirectional and just ensure net flow.

---

### 2. **Capacity Constraint on Edges**

For every edge $e$, total flow across that edge (summed over all demands) must be ≤ the installed capacity:

$$
\sum_{(s,t,d_{st}) \in D} x_e^{st} \leq \sum_{(e,u,m) \in M} u \cdot y_{e,m}
$$

> You can only push flow if the edge has enough installed capacity.

---

### 3. **Module Selection Constraint**

You can **install at most one module per edge** (or none):

$$
\sum_{(e,u,m) \in M} y_{e,m} \leq 1 \quad \forall e
$$

> You must pick only one capacity option per edge (if any).

---

## 🔚 Summary

So mathematically, you’ve built a **Mixed Integer Program** with:

* **Objective:** Minimize routing and module cost
* **Variables:**

  * Continuous flows $x_e^{st}$
  * Binary module decisions $y_{e,m}$
* **Constraints:**

  * Flow conservation per demand
  * Capacity limits per edge
  * Single module per edge

This is much richer and harder than MST — it captures flow, modular capacity, and cost simultaneously. The problem is NP-hard.

---

