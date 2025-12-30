from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import math
import random
import time
import statistics
import matplotlib.pyplot as plt

# ============================================================
# Global parameters for the Tabu Search
# ============================================================

MAX_ITERS = 1000                 # maximum TS iterations per run
TABU_TENURE = 20                 # number of iterations a move remains tabu
MAX_NEIGHBORS = 200              # neighbors evaluated per iteration

# ~6 minutes per run -> with 10 runs ~ 1 hour per instance
TIME_LIMIT_TS_RUN = 600         # time limit per TS run (seconds)

N_RUNS_PER_INSTANCE = 10         # number of runs per instance for statistics

# ============================================================
# Gurobi optimal objective values (from the project statement)
# ============================================================

OPTIMAL_OBJECTIVES: Dict[str, float] = {
    "FSTSP-40-1-1": 241.0,
    "FSTSP-40-1-2": 210.0,
    "FSTSP-60-1-1": 321.0,
    "FSTSP-60-1-2": 282.0,
    "FSTSP-60-2-1": 305.0,
    "FSTSP-60-2-2": 286.0,
    "FSTSP-80-1-1": 357.0,
    "FSTSP-80-1-2": 345.0,
    "FSTSP-80-2-1": 385.0,
    "FSTSP-80-2-2": 375.0,
}


# ============================================================
# Problem data structures and instance parsing
# ============================================================

@dataclass
class FSTSPInstance:
    n_nodes: int
    customers: List[int]
    depot_start: int
    depot_end: int
    truck_time: List[List[float]]
    drone_time: List[List[float]]
    endurance: float
    sl: float
    sr: float
    truck_only: List[bool]
    x: List[float]
    y: List[float]

    def is_valid_truck_arc(self, i: int, j: int) -> bool:
        """Return True if arc (i, j) is a valid truck arc with finite positive time."""
        if i == j:
            return False
        tij = self.truck_time[i][j]
        return tij > 0.0 and tij < math.inf

    def is_valid_drone_arc(self, i: int, j: int) -> bool:
        """Return True if arc (i, j) is a valid drone arc with finite positive time."""
        if i == j:
            return False
        dij = self.drone_time[i][j]
        return dij > 0.0 and dij < math.inf


def parse_instance_file(path: Path) -> FSTSPInstance:
    """
    Parse a benchmark instance in the format described in the project statement.
    """
    with path.open("r") as f:
        # First line: E SL SR
        line = f.readline()
        while line and line.strip() == "":
            line = f.readline()
        if not line:
            raise ValueError(f"Empty or invalid file: {path}")
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"First line must have E SL SR in {path}")
        endurance = float(parts[0])
        sl = float(parts[1])
        sr = float(parts[2])

        # Second line: |C|
        line = f.readline()
        while line and line.strip() == "":
            line = f.readline()
        if not line:
            raise ValueError(f"Missing line with |C| in {path}")
        C = int(line.split()[0])

        n = C + 2  # total number of nodes (customers + 2 depots)
        depot_start = 0
        depot_end = n - 1
        customers = list(range(1, n - 1))

        # Coordinates and truck_only flags
        truck_only = [False] * n
        xs: List[float] = [0.0] * n
        ys: List[float] = [0.0] * n

        # Depot start coordinates
        line = f.readline()
        if not line:
            raise ValueError(f"Missing depot start coordinates in {path}")
        parts = line.split()
        if len(parts) >= 2:
            xs[0] = float(parts[0])
            ys[0] = float(parts[1])

        # Customers coordinates and truck_only flags
        for idx in range(1, n - 1):
            line = f.readline()
            if not line:
                raise ValueError(
                    f"Unexpected end of file while reading customers in {path}"
                )
            parts = line.split()
            if len(parts) >= 2:
                xs[idx] = float(parts[0])
                ys[idx] = float(parts[1])
            if len(parts) >= 3:
                flag = int(parts[2])
                truck_only[idx] = (flag == 1)

        # Depot end coordinates
        line = f.readline()
        if not line:
            raise ValueError(f"Missing depot end coordinates in {path}")
        parts = line.split()
        if len(parts) >= 2:
            xs[n - 1] = float(parts[0])
            ys[n - 1] = float(parts[1])

        # Truck arcs
        raw_arcs_truck: List[Tuple[int, int, float]] = []
        for _ in range(n * n):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            i = int(parts[0])
            j = int(parts[1])
            t = float(parts[2])
            raw_arcs_truck.append((i, j, t))

        # Drone arcs
        raw_arcs_drone: List[Tuple[int, int, float]] = []
        for _ in range(n * n):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            i = int(parts[0])
            j = int(parts[1])
            d = float(parts[2])
            raw_arcs_drone.append((i, j, d))

    # Detect if arc indices are 0-based or 1-based
    all_idx = [i for (i, _, _) in raw_arcs_truck] + [j for (_, j, _) in raw_arcs_truck]
    if not all_idx:
        raise ValueError("No arcs read in " + str(path))
    min_idx = min(all_idx)
    max_idx = max(all_idx)

    if max_idx == n - 1 and min_idx == 0:
        def conv(i: int) -> int:
            return i
    elif max_idx == n and min_idx == 1:
        def conv(i: int) -> int:
            return i - 1
    else:
        # Fallback: assume they are already 0-based
        def conv(i: int) -> int:
            return i

    big = math.inf
    truck_time = [[big for _ in range(n)] for _ in range(n)]
    drone_time = [[big for _ in range(n)] for _ in range(n)]

    for i_raw, j_raw, t in raw_arcs_truck:
        i = conv(i_raw)
        j = conv(j_raw)
        if 0 <= i < n and 0 <= j < n and t > 0.0:
            truck_time[i][j] = float(t)

    for i_raw, j_raw, d in raw_arcs_drone:
        i = conv(i_raw)
        j = conv(j_raw)
        if 0 <= i < n and 0 <= j < n and d > 0.0:
            drone_time[i][j] = float(d)

    return FSTSPInstance(
        n_nodes=n,
        customers=customers,
        depot_start=depot_start,
        depot_end=depot_end,
        truck_time=truck_time,
        drone_time=drone_time,
        endurance=endurance,
        sl=sl,
        sr=sr,
        truck_only=truck_only,
        x=xs,
        y=ys,
    )


# ============================================================
# Solution representation and helpers
# ============================================================

@dataclass
class DroneSortie:
    """
    A drone sortie: launch at truck_route[L_index], serve customer h,
    recover at truck_route[R_index], with R_index > L_index.
    """
    L_index: int
    h: int
    R_index: int


def build_initial_perm(instance: FSTSPInstance) -> List[int]:
    """
    Build an initial permutation of all customers using a nearest-neighbor
    heuristic based on truck travel times.
    """
    unvisited = set(instance.customers)
    perm: List[int] = []
    current = instance.depot_start
    while unvisited:
        # Choose the nearest customer by truck distance
        next_c = min(unvisited, key=lambda c: instance.truck_time[current][c])
        perm.append(next_c)
        unvisited.remove(next_c)
        current = next_c
    return perm


def assign_sorties_greedy(instance: FSTSPInstance,
                          route: List[int]) -> Tuple[List[int], List[DroneSortie]]:
    """
    Given a truck route [s, ..., t] visiting all customers, try to remove some
    customers and serve them by drone using local patterns (L, h, R), where
    L and R become adjacent in the reduced truck route.

    We only consider "local" sorties of the form:
        truck: ... -> L -> h -> R -> ...
        drone:      L -> h -> R

    After creating the sortie, h is removed from the truck route.
    We enforce:
      - h is not truck-only,
      - d(L,h) + d(h,R) <= endurance,
      - truck_time(L,R) <= endurance,
      - the sortie yields a rough positive time saving (truck-only vs truck+drone).

    Returns:
        new_truck_route, sorties
    """
    sorties: List[DroneSortie] = []
    pos = 1  # first customer position in the route
    used_launch_from_depot = False

    while pos < len(route) - 1:
        i = route[pos - 1]
        h = route[pos]
        j = route[pos + 1]

        can_use = True

        # h cannot be truck-only
        if instance.truck_only[h]:
            can_use = False

        # Valid drone arcs i->h and h->j
        if can_use and not instance.is_valid_drone_arc(i, h):
            can_use = False
        if can_use and not instance.is_valid_drone_arc(h, j):
            can_use = False

        if can_use:
            d_Lh = instance.drone_time[i][h]
            d_hR = instance.drone_time[h][j]
            t_drone = d_Lh + d_hR
            # Drone endurance for pure flight
            if t_drone > instance.endurance:
                can_use = False

        if can_use:
            # Truck time between L and R (after removing h) is just t(i,j)
            t_truck_segment = instance.truck_time[i][j]
            if not (t_truck_segment > 0.0 and t_truck_segment < math.inf):
                can_use = False
            elif t_truck_segment > instance.endurance:
                can_use = False

        if can_use:
            # Rough benefit check: compare truck-only time with truck+drone
            t_old = instance.truck_time[i][h] + instance.truck_time[h][j]
            t_new = instance.truck_time[i][j]

            # First sortie from depot does not pay SL
            if i == instance.depot_start and not used_launch_from_depot:
                sl_eff = 0.0
            else:
                sl_eff = instance.sl

            approx_gain = t_old - (t_new + sl_eff + instance.sr)
            if approx_gain <= 0.0:
                can_use = False

        if can_use:
            # Accept sortie: indices are in the NEW route after removing h
            L_index = pos - 1
            # Remove h from the route: now j shifts to index 'pos'
            del route[pos]
            R_index = pos
            if i == instance.depot_start:
                used_launch_from_depot = True
            sorties.append(DroneSortie(L_index=L_index, h=h, R_index=R_index))
            # Do not increment pos, because the route shrank
            continue

        pos += 1

    return route, sorties


def simulate_route(instance: FSTSPInstance,
                   truck_route: List[int],
                   sorties: List[DroneSortie]) -> float:
    """
    Simulate the execution of the truck and a single drone along the given
    truck_route with the specified sorties.

    The objective value is the total time until the truck returns to the final depot,
    including:
      - truck travel times,
      - launch times SL (except the very first sortie if it starts from the depot),
      - recovery times SR,
      - any waiting time of the truck at recovery nodes.

    We enforce:
      - each customer is served exactly once (truck OR drone),
      - truck-only customers cannot be served by the drone,
      - no overlapping sorties,
      - drone endurance for each sortie.

    If the solution is infeasible, returns math.inf.
    """
    n_edges = len(truck_route) - 1
    if n_edges < 1:
        return math.inf

    # Map launch and recovery indices to sorties (assume no overlaps)
    launch_map: Dict[int, DroneSortie] = {}
    recovery_map: Dict[int, DroneSortie] = {}
    for srt in sorties:
        if srt.L_index in launch_map or srt.R_index in recovery_map:
            # overlapping or conflicting sorties in terms of indices
            return math.inf
        launch_map[srt.L_index] = srt
        recovery_map[srt.R_index] = srt

    # Check that every customer is served exactly once (truck or drone)
    served_by_truck = set(truck_route[1:-1])  # interior nodes of truck route
    served_by_drone = {srt.h for srt in sorties}
    all_served = served_by_truck | served_by_drone
    if set(instance.customers) != all_served:
        return math.inf
    if served_by_truck & served_by_drone:
        # intersection must be empty
        return math.inf
    # Truck-only customers cannot be served by drone
    for h in served_by_drone:
        if instance.truck_only[h]:
            return math.inf

    time_global = 0.0
    drone_active = False
    drone_start_time = 0.0
    drone_end_flight_time = 0.0
    depot_sortie_free_used = False

    for edge_idx in range(n_edges):
        i = truck_route[edge_idx]
        j = truck_route[edge_idx + 1]

        # (1) Launch drone if there is a sortie starting here
        if edge_idx in launch_map:
            if drone_active:
                # cannot launch a new sortie while another is active
                return math.inf
            srt = launch_map[edge_idx]
            L_node = truck_route[srt.L_index]
            R_node = truck_route[srt.R_index]
            h = srt.h

            # Consistency check: launch and recovery nodes must match the route
            if L_node != i or R_node != truck_route[edge_idx + 1]:
                return math.inf

            # Launch service time: first sortie from depot does not pay SL
            if L_node == instance.depot_start and not depot_sortie_free_used:
                launch_service = 0.0
                depot_sortie_free_used = True
            else:
                launch_service = instance.sl

            time_global += launch_service
            drone_start_time = time_global

            d_Lh = instance.drone_time[L_node][h]
            d_hR = instance.drone_time[h][R_node]
            if not (d_Lh > 0.0 and d_Lh < math.inf and
                    d_hR > 0.0 and d_hR < math.inf):
                return math.inf

            drone_end_flight_time = drone_start_time + d_Lh + d_hR

            # Drone endurance for pure flight
            if (d_Lh + d_hR) > instance.endurance:
                return math.inf

            drone_active = True

        # (2) Truck moves i -> j
        tij = instance.truck_time[i][j]
        if not (tij > 0.0 and tij < math.inf):
            return math.inf
        time_global += tij

        # (3) If a sortie recovers here
        if drone_active and (edge_idx + 1) in recovery_map:
            # truck arrival at the recovery node
            truck_arrival_R = time_global

            # drone arrival at R after finishing its flight
            if drone_end_flight_time > truck_arrival_R:
                # Truck waits for the drone
                wait = drone_end_flight_time - truck_arrival_R
                time_global += wait
                truck_arrival_R = time_global

            # total airborne time = from drone_start_time to meeting with the truck
            airborne_time = truck_arrival_R - drone_start_time
            if airborne_time > instance.endurance:
                return math.inf

            # Recovery service time
            time_global += instance.sr
            drone_active = False

    # At the end, drone must be back on the truck
    if drone_active:
        return math.inf

    return time_global


def evaluate_perm(instance: FSTSPInstance,
                  perm: List[int]) -> Tuple[float, List[int], List[DroneSortie]]:
    """
    From a permutation of all customers, build the base truck route,
    decide drone sorties with the greedy procedure, and simulate the resulting plan.

    Returns:
        cost, final_truck_route, sorties
    """
    base_route = [instance.depot_start] + perm + [instance.depot_end]
    route_truck = list(base_route)
    route_truck, sorties = assign_sorties_greedy(instance, route_truck)
    cost = simulate_route(instance, route_truck, sorties)
    return cost, route_truck, sorties


# ============================================================
# Plotting helpers (with saving + 30s timeout)
# ============================================================

def _show_and_save(fig: plt.Figure,
                   save_path: Optional[Path],
                   show_seconds: float = 30.0) -> None:
    """
    Helper: save figure (if path provided), show non-blocking, keep up to
    show_seconds and then close it automatically.
    """
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")

    # Show non-blocking and close after timeout
    plt.show(block=False)
    plt.pause(show_seconds)
    plt.close(fig)


def plot_route_on_axes(ax,
                       instance: FSTSPInstance,
                       truck_route: List[int],
                       sorties: List[DroneSortie],
                       title: str,
                       highlight_best: bool) -> None:
    """
    Draw a single route (truck + drone sorties) on the provided Axes.
    If highlight_best is True, draw the truck route with thicker/darker lines.
    """
    x = instance.x
    y = instance.y

    # Customers
    customers = instance.customers
    cx = [x[i] for i in customers]
    cy = [y[i] for i in customers]
    ax.scatter(cx, cy, s=10, c="lightgray")

    # Truck-only customers
    tx_only = [i for i in customers if instance.truck_only[i]]
    if tx_only:
        tox = [x[i] for i in tx_only]
        toy = [y[i] for i in tx_only]
        ax.scatter(tox, toy, s=20, marker="s", edgecolors="k",
                   facecolors="none")

    # Depots
    ax.scatter([x[instance.depot_start]], [y[instance.depot_start]],
               s=40, c="green")
    ax.scatter([x[instance.depot_end]], [y[instance.depot_end]],
               s=40, c="black")

    # Truck route
    truck_color = "darkred" if highlight_best else "red"
    truck_width = 2.5 if highlight_best else 1.5
    for i in range(len(truck_route) - 1):
        u = truck_route[i]
        v = truck_route[i + 1]
        ax.plot([x[u], x[v]], [y[u], y[v]],
                linewidth=truck_width, color=truck_color)

    # Drone sorties
    for srt in sorties:
        L_node = truck_route[srt.L_index]
        R_node = truck_route[srt.R_index]
        h = srt.h
        ax.plot([x[L_node], x[h]], [y[L_node], y[h]],
                linestyle="--", linewidth=1.0, color="cyan")
        ax.plot([x[h], x[R_node]], [y[h], y[R_node]],
                linestyle="--", linewidth=1.0, color="cyan")

    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal", "box")
    ax.tick_params(labelsize=6)


def plot_all_routes_subplots(instance_name: str,
                             instance: FSTSPInstance,
                             routes: List[List[int]],
                             sorties_list: List[List[DroneSortie]],
                             costs: List[float],
                             save_path: Optional[Path] = None,
                             show_seconds: float = 30.0) -> None:
    """
    Build a single figure with subplots for all runs.
    Each subplot shows the route of one run (truck + drone).
    The best run (minimum cost) is highlighted.
    """
    n_runs = len(routes)
    if n_runs == 0:
        return

    # Pick best run among finite costs
    finite_indices = [i for i, c in enumerate(costs) if not math.isinf(c)]
    if finite_indices:
        best_idx = min(finite_indices, key=lambda i: costs[i])
    else:
        best_idx = None

    cols = min(n_runs, 5)
    rows = math.ceil(n_runs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx in range(n_runs):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        route = routes[idx]
        sorties = sorties_list[idx]
        cost = costs[idx]

        if not route or math.isinf(cost):
            ax.text(0.5, 0.5,
                    f"Run {idx + 1}\nNo feasible\nsolution",
                    ha="center", va="center", fontsize=8)
            ax.set_axis_off()
            continue

        is_best = (best_idx is not None and idx == best_idx)
        title = f"Run {idx + 1} (cost={cost:.1f})"
        if is_best:
            title += " [BEST]"

        plot_route_on_axes(ax, instance, route, sorties, title, highlight_best=is_best)

    # Turn off any extra axes if n_runs < rows*cols
    for idx in range(n_runs, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].set_axis_off()

    fig.suptitle(f"All routes for {instance_name} (best highlighted)", fontsize=14)
    fig.tight_layout()

    _show_and_save(fig, save_path, show_seconds)


# --------- separate bar plots for cost / time / expansions ---------

def plot_run_costs_bars(instance_name: str,
                        costs: List[float],
                        run_numbers: List[int],
                        save_path: Optional[Path] = None,
                        show_seconds: float = 30.0) -> None:
    """Bar chart: cost per run."""
    x_pos = list(range(len(run_numbers)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_pos, costs)
    ax.set_ylabel("Cost")
    ax.set_title(f"Costs per run - {instance_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Run index")
    fig.tight_layout()

    _show_and_save(fig, save_path, show_seconds)


def plot_run_times_bars(instance_name: str,
                        times: List[float],
                        run_numbers: List[int],
                        save_path: Optional[Path] = None,
                        show_seconds: float = 30.0) -> None:
    """Bar chart: runtime per run."""
    x_pos = list(range(len(run_numbers)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_pos, times)
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtimes per run - {instance_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Run index")
    fig.tight_layout()

    _show_and_save(fig, save_path, show_seconds)


def plot_run_expansions_bars(instance_name: str,
                             expansions: List[int],
                             run_numbers: List[int],
                             save_path: Optional[Path] = None,
                             show_seconds: float = 30.0) -> None:
    """Bar chart: expansions per run."""
    x_pos = list(range(len(run_numbers)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_pos, expansions)
    ax.set_ylabel("Expansions")
    ax.set_title(f"Expansions per run - {instance_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Run index")
    fig.tight_layout()

    _show_and_save(fig, save_path, show_seconds)


def plot_run_customers_bars(instance_name: str,
                            truck_customers: List[int],
                            drone_customers: List[int],
                            run_numbers: List[int],
                            save_path: Optional[Path] = None,
                            show_seconds: float = 30.0) -> None:
    """
    Bar chart: number of customers served by truck and by drone per run.
    """
    if not run_numbers:
        return

    x_pos = list(range(len(run_numbers)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([x - width / 2 for x in x_pos], truck_customers,
           width=width, label="Truck customers")
    ax.bar([x + width / 2 for x in x_pos], drone_customers,
           width=width, label="Drone customers")

    ax.set_ylabel("Number of customers")
    ax.set_title(f"Customers served per run - {instance_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Run index")
    ax.legend()
    fig.tight_layout()

    _show_and_save(fig, save_path, show_seconds)


# --------- global comparison plots vs Gurobi (GAP and ratio) ---------

def plot_gaps_per_instance(instance_keys: List[str],
                           gaps_best: List[float],
                           gaps_avg: List[float],
                           save_path: Optional[Path] = None,
                           show_seconds: float = 30.0) -> None:
    """
    Bar chart of GAP(best) and GAP(avg) vs Gurobi optimal, one pair of bars per instance.
    """
    if not instance_keys:
        return

    x_pos = list(range(len(instance_keys)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([x - width / 2 for x in x_pos], gaps_best,
           width=width, label="GAP(best)")
    ax.bar([x + width / 2 for x in x_pos], gaps_avg,
           width=width, label="GAP(avg)")

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("GAP (%)")
    ax.set_title("GAP vs Gurobi optimal per instance")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(instance_keys, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    _show_and_save(fig, save_path, show_seconds)


def plot_ratio_best_per_instance(instance_keys: List[str],
                                 ratios_best: List[float],
                                 save_path: Optional[Path] = None,
                                 show_seconds: float = 30.0) -> None:
    """
    Bar chart of best_cost / optimal_cost per instance.
    A value close to 1.0 means very close to Gurobi's solution.
    """
    if not instance_keys:
        return

    x_pos = list(range(len(instance_keys)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos, ratios_best)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5,
               label="Optimal ratio = 1.0")

    ax.set_ylabel("best_cost / optimal_cost")
    ax.set_title("Best-to-optimal cost ratio per instance")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(instance_keys, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    _show_and_save(fig, save_path, show_seconds)


# ============================================================
# Tabu Search metaheuristic
# ============================================================

def tabu_search(instance: FSTSPInstance,
                max_iters: int = MAX_ITERS,
                tabu_tenure: int = TABU_TENURE,
                max_neighbors: int = MAX_NEIGHBORS,
                time_limit_seconds: float = TIME_LIMIT_TS_RUN
                ) -> Tuple[float, List[int], List[int], List[DroneSortie], int]:
    """
    Tabu Search on permutations of customers.
    - State: a permutation perm of all customers.
    - Neighborhood: swap moves (perm[i] <-> perm[j]) for random pairs (i, j).
    - Evaluation: build truck route + greedy drone sorties, then simulate.

    Returns:
        final_cost, best_perm, best_truck_route, best_sorties, expansions

    where 'expansions' is the total number of neighbor evaluations
    (calls to evaluate_perm) performed in this run.
    """
    start_time = time.time()
    expansions = 0

    # ----- Initial solution -----
    current_perm = build_initial_perm(instance)
    current_cost, _, _ = evaluate_perm(instance, current_perm)
    expansions += 1

    # If somehow infeasible, try random restarts
    tries = 0
    while math.isinf(current_cost) and tries < 20:
        random.shuffle(current_perm)
        current_cost, _, _ = evaluate_perm(instance, current_perm)
        expansions += 1
        tries += 1

    # If still infeasible, abort
    if math.isinf(current_cost):
        return math.inf, current_perm, [], [], expansions

    best_perm = current_perm[:]
    best_cost = current_cost

    # Tabu list: (customer_a, customer_b) -> expiry iteration
    tabu: Dict[Tuple[int, int], int] = {}
    iteration = 0

    # ----- Main Tabu Search loop -----
    while iteration < max_iters and (time.time() - start_time) < time_limit_seconds:
        iteration += 1
        best_candidate_perm: Optional[List[int]] = None
        best_candidate_cost = math.inf
        best_candidate_move: Optional[Tuple[int, int]] = None

        n = len(current_perm)
        if n < 2:
            break

        # Generate and evaluate neighbors
        for _ in range(max_neighbors):
            i = random.randrange(0, n)
            j = random.randrange(0, n)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            a = current_perm[i]
            b = current_perm[j]
            move_sig = (min(a, b), max(a, b))

            # Apply swap
            neighbor_perm = current_perm[:]
            neighbor_perm[i], neighbor_perm[j] = neighbor_perm[j], neighbor_perm[i]

            cost, _, _ = evaluate_perm(instance, neighbor_perm)
            expansions += 1

            if math.isinf(cost):
                # infeasible neighbor, skip
                continue

            is_tabu = move_sig in tabu and tabu[move_sig] > iteration
            # Aspiration: allow tabu move if it improves global best
            if is_tabu and cost >= best_cost:
                continue

            if cost < best_candidate_cost:
                best_candidate_cost = cost
                best_candidate_perm = neighbor_perm
                best_candidate_move = move_sig

        # If no feasible neighbor found, restart from a new heuristic solution
        if best_candidate_perm is None:
            current_perm = build_initial_perm(instance)
            current_cost, _, _ = evaluate_perm(instance, current_perm)
            expansions += 1
            if math.isinf(current_cost):
                break
            continue

        # Move to the best candidate
        current_perm = best_candidate_perm
        current_cost = best_candidate_cost

        # Update tabu list
        if best_candidate_move is not None:
            tabu[best_candidate_move] = iteration + tabu_tenure

        # Clean expired tabu entries occasionally
        if iteration % 50 == 0:
            expired_keys = [k for k, exp in tabu.items() if exp <= iteration]
            for k in expired_keys:
                del tabu[k]

        # Update global best
        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = current_perm[:]

    # Recompute detailed plan for the best permutation
    final_cost, best_truck_route, best_sorties = evaluate_perm(instance, best_perm)
    expansions += 1
    return final_cost, best_perm, best_truck_route, best_sorties, expansions


# ============================================================
# Running the heuristic on all benchmark instances
# ============================================================

def run_all_instances(n_runs_per_instance: int = N_RUNS_PER_INSTANCE) -> None:
    """
    Run the Tabu Search heuristic on all FSTSP-*.txt instances
    found in the ./instances directory, perform multiple runs
    per instance, print statistics, and plot results.

    For each instance:
      - run Tabu Search 'n_runs_per_instance' times,
      - keep the best solution,
      - build bar charts (cost, runtime, expansions, customers served),
      - build ONE figure with subplots, each subplot being one route,
        and compute the gap vs Gurobi optimal if available.

    At the end, build global comparison plots vs Gurobi (GAP and ratio),
    and create a 'report' folder with the plots and a CSV summary table.

    All plots are saved under:
      ./results/<instance_key>/        for per-instance plots
      ./results/report/                for report figures and table
    And each figure window stays at most 30 seconds open.
    """
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        # For interactive environments (e.g. notebooks)
        base_dir = Path().resolve()

    instances_dir = base_dir / "instances"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # New: directory for report material (plots + table)
    report_dir = results_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(instances_dir.glob("FSTSP-*.txt"))

    if not txt_files:
        print(f"No instances found in {instances_dir}")
        return

    # To store per-instance GAP and ratio information (for global plots)
    instances_with_opt: List[str] = []
    gaps_best_all: List[float] = []
    gaps_avg_all: List[float] = []
    ratios_best_all: List[float] = []

    # To store summary table data for the report
    summary_rows: List[Dict[str, float]] = []

    for path in txt_files:
        instance_name = path.name        # e.g. "FSTSP-40-1-1.txt"
        instance_key = path.stem         # e.g. "FSTSP-40-1-1"
        print(f"Solving instance {instance_name} with Tabu Search ({n_runs_per_instance} runs)...")
        instance = parse_instance_file(path)

        instance_results_dir = results_dir / instance_key
        instance_results_dir.mkdir(parents=True, exist_ok=True)

        all_costs: List[float] = []
        run_times: List[float] = []
        run_expansions: List[int] = []
        run_truck_routes: List[List[int]] = []
        run_sorties_list: List[List[DroneSortie]] = []

        best_overall_cost = math.inf
        best_overall_perm: Optional[List[int]] = None
        best_overall_truck_route: List[int] = []
        best_overall_sorties: List[DroneSortie] = []

        t0_inst = time.time()

        for run_idx in range(1, n_runs_per_instance + 1):
            # Different random seed per run (for reproducibility if desired)
            random.seed(run_idx)

            t0_run = time.time()
            cost, perm, truck_route, sorties, expansions = tabu_search(instance)
            t1_run = time.time()
            runtime_run = t1_run - t0_run

            all_costs.append(cost)
            run_times.append(runtime_run)
            run_expansions.append(expansions)
            run_truck_routes.append(truck_route)
            run_sorties_list.append(sorties)

            print(f"  Run {run_idx:2d}: cost = {cost:.2f}, "
                  f"time = {runtime_run:.2f} s, expansions = {expansions}")

            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_perm = perm[:]
                best_overall_truck_route = truck_route[:]
                best_overall_sorties = list(sorties)

        t1_inst = time.time()
        inst_runtime = t1_inst - t0_inst

        print(f"\nInstance {instance_name} summary:")
        if all(math.isinf(c) for c in all_costs):
            print("  No feasible solution found in any run.")
            print(f"  Total time for {n_runs_per_instance} runs: {inst_runtime:.2f} seconds\n")
            continue

        # Consider only finite-cost runs for statistics and bar plots
        finite_indices = [i for i, c in enumerate(all_costs) if not math.isinf(c)]
        finite_costs = [all_costs[i] for i in finite_indices]
        finite_times = [run_times[i] for i in finite_indices]
        finite_expansions = [run_expansions[i] for i in finite_indices]
        finite_run_numbers = [i + 1 for i in finite_indices]

        best_cost = min(finite_costs)
        avg_cost = statistics.mean(finite_costs)
        std_cost = statistics.pstdev(finite_costs) if len(finite_costs) > 1 else 0.0

        avg_time = statistics.mean(finite_times)
        std_time = statistics.pstdev(finite_times) if len(finite_times) > 1 else 0.0

        avg_exp = statistics.mean(finite_expansions)
        std_exp = statistics.pstdev(finite_expansions) if len(finite_expansions) > 1 else 0.0

        # Customers served by truck and by drone per run (finite runs only)
        truck_customers_counts: List[int] = []
        drone_customers_counts: List[int] = []
        for idx in finite_indices:
            route = run_truck_routes[idx]
            sorties_run = run_sorties_list[idx]

            if not route:
                truck_customers_counts.append(0)
                drone_customers_counts.append(0)
                continue

            drone_customers_run = {s.h for s in sorties_run}
            # Customers in the truck route, excluding depots and excluding drone-served ones
            truck_customers_run = [
                c for c in route[1:-1] if c not in drone_customers_run
            ]

            truck_customers_counts.append(len(truck_customers_run))
            drone_customers_counts.append(len(drone_customers_run))

        print(f"  Best cost over runs: {best_cost:.2f}")
        print(f"  Average cost (finite runs): {avg_cost:.2f}")
        print(f"  Std. dev. of cost: {std_cost:.2f}")
        print(f"  Average runtime per run (finite): {avg_time:.2f} s")
        print(f"  Std. dev. runtime: {std_time:.2f} s")
        print(f"  Average expansions per run (finite): {avg_exp:.1f}")
        print(f"  Std. dev. expansions: {std_exp:.1f}")
        print(f"  Total time for {n_runs_per_instance} runs: {inst_runtime:.2f} seconds")

        # ===== GAP vs Gurobi optimal =====
        opt_val = OPTIMAL_OBJECTIVES.get(instance_key, math.nan)
        gap_best = math.nan
        gap_avg = math.nan
        ratio_best = math.nan

        if instance_key in OPTIMAL_OBJECTIVES:
            gap_best = 100.0 * (best_cost - opt_val) / opt_val
            gap_avg = 100.0 * (avg_cost - opt_val) / opt_val
            ratio_best = best_cost / opt_val

            print(f"  Gurobi optimal objective: {opt_val:.2f}")
            print(f"  GAP(best)  = {gap_best:.2f}%")
            print(f"  GAP(avg)   = {gap_avg:.2f}%")

            # Store for global comparison plots across instances
            instances_with_opt.append(instance_key)
            gaps_best_all.append(gap_best)
            gaps_avg_all.append(gap_avg)
            ratios_best_all.append(ratio_best)
        else:
            print("  [INFO] No Gurobi optimal value stored for this instance; gaps not computed.")

        # Row for the summary table (report)
        avg_truck_customers = statistics.mean(truck_customers_counts) if truck_customers_counts else 0.0
        avg_drone_customers = statistics.mean(drone_customers_counts) if drone_customers_counts else 0.0

        summary_rows.append({
            "instance": instance_key,
            "opt_value": opt_val if not math.isnan(opt_val) else "",
            "best_cost": best_cost,
            "avg_cost": avg_cost,
            "std_cost": std_cost,
            "gap_best": gap_best if not math.isnan(gap_best) else "",
            "gap_avg": gap_avg if not math.isnan(gap_avg) else "",
            "ratio_best": ratio_best if not math.isnan(ratio_best) else "",
            "avg_time": avg_time,
            "std_time": std_time,
            "avg_exp": avg_exp,
            "std_exp": std_exp,
            "avg_truck_customers": avg_truck_customers,
            "avg_drone_customers": avg_drone_customers,
        })

        if best_overall_perm is not None:
            drone_customers = {s.h for s in best_overall_sorties}
            truck_customers = best_overall_truck_route[1:-1]
            print(f"  Truck route length (including depots): {len(best_overall_truck_route)}")
            print(f"  # truck-served customers: {len(truck_customers)}")
            print(f"  # drone-served customers: {len(drone_customers)}")
            print(f"  # drone sorties: {len(best_overall_sorties)}")

            # Separate bar plots for costs, times and expansions (finite runs only)
            plot_run_costs_bars(
                instance_name,
                finite_costs,
                finite_run_numbers,
                save_path=instance_results_dir / "costs_per_run.png",
                show_seconds=30.0,
            )
            plot_run_times_bars(
                instance_name,
                finite_times,
                finite_run_numbers,
                save_path=instance_results_dir / "times_per_run.png",
                show_seconds=30.0,
            )
            plot_run_expansions_bars(
                instance_name,
                finite_expansions,
                finite_run_numbers,
                save_path=instance_results_dir / "expansions_per_run.png",
                show_seconds=30.0,
            )

            # New plot: customers served by truck and by drone per run
            plot_run_customers_bars(
                instance_name,
                truck_customers_counts,
                drone_customers_counts,
                finite_run_numbers,
                save_path=instance_results_dir / "customers_per_run.png",
                show_seconds=30.0,
            )

            # Single plot with subplots for ALL routes (one per run),
            # highlighting the best one.
            plot_all_routes_subplots(
                instance_name,
                instance,
                run_truck_routes,
                run_sorties_list,
                all_costs,
                save_path=instance_results_dir / "routes_all_runs.png",
                show_seconds=30.0,
            )

        print("\n" + "=" * 60 + "\n")

    # ===== Global comparison plots vs Gurobi across instances =====
    if instances_with_opt:
        # Save global plots directly into the 'report' folder
        plot_gaps_per_instance(
            instances_with_opt,
            gaps_best_all,
            gaps_avg_all,
            save_path=report_dir / "gaps_vs_gurobi.png",
            show_seconds=30.0,
        )
        plot_ratio_best_per_instance(
            instances_with_opt,
            ratios_best_all,
            save_path=report_dir / "ratio_vs_gurobi.png",
            show_seconds=30.0,
        )

    # ===== Summary table for the report (CSV) =====
    if summary_rows:
        table_path = report_dir / "summary_table_report.csv"
        header = [
            "instance",
            "opt_value",
            "best_cost",
            "avg_cost",
            "std_cost",
            "gap_best",
            "gap_avg",
            "ratio_best",
            "avg_time",
            "std_time",
            "avg_exp",
            "std_exp",
            "avg_truck_customers",
            "avg_drone_customers",
        ]
        with table_path.open("w") as f:
            f.write(",".join(header) + "\n")
            for row in summary_rows:
                values = [str(row[h]) for h in header]
                f.write(",".join(values) + "\n")

        print(f"Summary table for report saved to: {table_path}")


if __name__ == "__main__":
    run_all_instances()
