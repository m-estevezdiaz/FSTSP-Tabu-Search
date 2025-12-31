from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import math
import random
import time
import statistics
import matplotlib.pyplot as plt

# ============================================================
# Global parameters (Tabu Search)
# ============================================================

MAX_ITERS = 800
TABU_TENURE = 20
MAX_NEIGHBORS = 150

# Time limit per TS run (seconds)
TIME_LIMIT_TS_RUN = 500

# Number of runs per instance for statistics
N_RUNS_PER_INSTANCE = 5

# ============================================================
# Plot behavior (single window, no auto-close during run)
# ============================================================

PLOT_WINDOW_NUM = 1
PLOT_REFRESH_PAUSE = 0.001  # small pause so GUI can process events


# ============================================================
# Gurobi optimal objective values (given by the project)
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
        if i == j:
            return False
        tij = self.truck_time[i][j]
        return tij > 0.0 and tij < math.inf

    def is_valid_drone_arc(self, i: int, j: int) -> bool:
        if i == j:
            return False
        dij = self.drone_time[i][j]
        return dij > 0.0 and dij < math.inf


def parse_instance_file(path: Path) -> FSTSPInstance:
    """
    Parse a benchmark instance.
    Expected structure:
      - First line: E SL SR
      - Second line: |C|
      - Coordinates for depot start, customers, depot end
      - Truck arcs (n*n lines)
      - Drone arcs (n*n lines)
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

        n = C + 2
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
                raise ValueError(f"Unexpected end of file while reading customers in {path}")
            parts = line.split()
            if len(parts) >= 2:
                xs[idx] = float(parts[0])
                ys[idx] = float(parts[1])
            if len(parts) >= 3:
                truck_only[idx] = (int(parts[2]) == 1)

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
            raw_arcs_truck.append((int(parts[0]), int(parts[1]), float(parts[2])))

        # Drone arcs
        raw_arcs_drone: List[Tuple[int, int, float]] = []
        for _ in range(n * n):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            raw_arcs_drone.append((int(parts[0]), int(parts[1]), float(parts[2])))

    # Detect if indices are 0-based or 1-based
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
    A drone sortie:
      - launch at truck_route[L_index]
      - serve customer h
      - recover at truck_route[R_index]
    """
    L_index: int
    h: int
    R_index: int


def build_initial_perm(instance: FSTSPInstance) -> List[int]:
    """
    Initial permutation (nearest-neighbor using truck times).
    """
    unvisited = set(instance.customers)
    perm: List[int] = []
    current = instance.depot_start
    while unvisited:
        next_c = min(unvisited, key=lambda c: instance.truck_time[current][c])
        perm.append(next_c)
        unvisited.remove(next_c)
        current = next_c
    return perm


def assign_sorties_greedy(instance: FSTSPInstance, route: List[int]) -> Tuple[List[int], List[DroneSortie]]:
    """
    Greedy local sorties:
      Truck pattern: ... -> L -> h -> R -> ...
      Drone pattern:      L -> h -> R
    After accepting, remove h from the truck route.
    """
    sorties: List[DroneSortie] = []
    pos = 1
    used_launch_from_depot = False

    while pos < len(route) - 1:
        L = route[pos - 1]
        h = route[pos]
        R = route[pos + 1]

        can_use = True

        if instance.truck_only[h]:
            can_use = False

        if can_use and not instance.is_valid_drone_arc(L, h):
            can_use = False
        if can_use and not instance.is_valid_drone_arc(h, R):
            can_use = False

        if can_use:
            t_drone = instance.drone_time[L][h] + instance.drone_time[h][R]
            if t_drone > instance.endurance:
                can_use = False

        if can_use:
            t_truck_LR = instance.truck_time[L][R]
            if not (t_truck_LR > 0.0 and t_truck_LR < math.inf):
                can_use = False
            elif t_truck_LR > instance.endurance:
                can_use = False

        if can_use:
            t_old = instance.truck_time[L][h] + instance.truck_time[h][R]
            t_new = instance.truck_time[L][R]

            if L == instance.depot_start and not used_launch_from_depot:
                sl_eff = 0.0
            else:
                sl_eff = instance.sl

            approx_gain = t_old - (t_new + sl_eff + instance.sr)
            if approx_gain <= 0.0:
                can_use = False

        if can_use:
            L_index = pos - 1
            del route[pos]
            R_index = pos
            if L == instance.depot_start:
                used_launch_from_depot = True
            sorties.append(DroneSortie(L_index=L_index, h=h, R_index=R_index))
            continue

        pos += 1

    return route, sorties


def simulate_route(instance: FSTSPInstance, truck_route: List[int], sorties: List[DroneSortie]) -> float:
    """
    Simulate truck + single drone:
      - add SL at launch (except first depot launch)
      - add truck travel times
      - truck waits at recovery if needed
      - add SR at recovery
      - enforce endurance on airborne time
    Return total completion time, or inf if infeasible.
    """
    n_edges = len(truck_route) - 1
    if n_edges < 1:
        return math.inf

    launch_map: Dict[int, DroneSortie] = {}
    recovery_map: Dict[int, DroneSortie] = {}

    for srt in sorties:
        if srt.L_index in launch_map or srt.R_index in recovery_map:
            return math.inf
        launch_map[srt.L_index] = srt
        recovery_map[srt.R_index] = srt

    served_by_truck = set(truck_route[1:-1])
    served_by_drone = {srt.h for srt in sorties}
    all_served = served_by_truck | served_by_drone

    if set(instance.customers) != all_served:
        return math.inf
    if served_by_truck & served_by_drone:
        return math.inf
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

        # Launch
        if edge_idx in launch_map:
            if drone_active:
                return math.inf

            srt = launch_map[edge_idx]
            L_node = truck_route[srt.L_index]
            R_node = truck_route[srt.R_index]
            h = srt.h

            if L_node != i:
                return math.inf
            if R_node != truck_route[srt.R_index]:
                return math.inf

            if L_node == instance.depot_start and not depot_sortie_free_used:
                launch_service = 0.0
                depot_sortie_free_used = True
            else:
                launch_service = instance.sl

            time_global += launch_service
            drone_start_time = time_global

            d_Lh = instance.drone_time[L_node][h]
            d_hR = instance.drone_time[h][R_node]
            if not (d_Lh > 0.0 and d_Lh < math.inf and d_hR > 0.0 and d_hR < math.inf):
                return math.inf

            if (d_Lh + d_hR) > instance.endurance:
                return math.inf

            drone_end_flight_time = drone_start_time + d_Lh + d_hR
            drone_active = True

        # Truck move
        tij = instance.truck_time[i][j]
        if not (tij > 0.0 and tij < math.inf):
            return math.inf
        time_global += tij

        # Recovery
        if drone_active and (edge_idx + 1) in recovery_map:
            truck_arrival_R = time_global

            if drone_end_flight_time > truck_arrival_R:
                time_global += (drone_end_flight_time - truck_arrival_R)
                truck_arrival_R = time_global

            airborne_time = truck_arrival_R - drone_start_time
            if airborne_time > instance.endurance:
                return math.inf

            time_global += instance.sr
            drone_active = False

    if drone_active:
        return math.inf

    return time_global


def evaluate_perm(instance: FSTSPInstance, perm: List[int]) -> Tuple[float, List[int], List[DroneSortie]]:
    """
    From a permutation:
      1) build base truck route
      2) greedy sorties
      3) simulate
    """
    base_route = [instance.depot_start] + perm + [instance.depot_end]
    route_truck = list(base_route)
    route_truck, sorties = assign_sorties_greedy(instance, route_truck)
    cost = simulate_route(instance, route_truck, sorties)
    return cost, route_truck, sorties


# ============================================================
# Plot helpers (single window, refresh only)
# ============================================================

def _get_single_figure(figsize: Tuple[float, float]) -> plt.Figure:
    plt.figure(num=PLOT_WINDOW_NUM)
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1], forward=True)
    return fig


def _show_and_save(fig: plt.Figure, save_path: Optional[Path]) -> None:
    """
    Save (optional) and refresh the same interactive window.
    """
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")

    plt.show(block=False)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(PLOT_REFRESH_PAUSE)


def plot_route_on_axes(
    ax,
    instance: FSTSPInstance,
    truck_route: List[int],
    sorties: List[DroneSortie],
    title: str,
    highlight_best: bool
) -> None:
    """
    Draw truck route + drone sorties on a given axes.
    """
    x = instance.x
    y = instance.y

    customers = instance.customers
    ax.scatter([x[i] for i in customers], [y[i] for i in customers], s=10, c="lightgray")

    truck_only_nodes = [i for i in customers if instance.truck_only[i]]
    if truck_only_nodes:
        ax.scatter(
            [x[i] for i in truck_only_nodes],
            [y[i] for i in truck_only_nodes],
            s=20,
            marker="s",
            edgecolors="k",
            facecolors="none",
        )

    ax.scatter([x[instance.depot_start]], [y[instance.depot_start]], s=40, c="green")
    ax.scatter([x[instance.depot_end]], [y[instance.depot_end]], s=40, c="black")

    truck_color = "darkred" if highlight_best else "red"
    truck_width = 2.5 if highlight_best else 1.5
    for i in range(len(truck_route) - 1):
        u = truck_route[i]
        v = truck_route[i + 1]
        ax.plot([x[u], x[v]], [y[u], y[v]], linewidth=truck_width, color=truck_color)

    for srt in sorties:
        L_node = truck_route[srt.L_index]
        R_node = truck_route[srt.R_index]
        h = srt.h
        ax.plot([x[L_node], x[h]], [y[L_node], y[h]], linestyle="--", linewidth=1.0, color="cyan")
        ax.plot([x[h], x[R_node]], [y[h], y[R_node]], linestyle="--", linewidth=1.0, color="cyan")

    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal", "box")
    ax.tick_params(labelsize=6)


def plot_all_routes_subplots(
    instance_name: str,
    instance: FSTSPInstance,
    routes: List[List[int]],
    sorties_list: List[List[DroneSortie]],
    costs: List[float],
    save_path: Optional[Path] = None
) -> None:
    """
    One figure with subplots showing all runs; best run highlighted.
    """
    n_runs = len(routes)
    if n_runs == 0:
        return

    finite_indices = [i for i, c in enumerate(costs) if not math.isinf(c)]
    best_idx = min(finite_indices, key=lambda i: costs[i]) if finite_indices else None

    cols = min(n_runs, 5)
    rows = math.ceil(n_runs / cols)

    FIXED_WIDTH = 8.0         
    ROW_HEIGHT  = 4.0          
    fig = _get_single_figure((FIXED_WIDTH, ROW_HEIGHT * rows))
    axes = plt.subplots(rows, cols, num=PLOT_WINDOW_NUM)[1]

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
            ax.text(0.5, 0.5, f"Run {idx + 1}\nNo feasible\nsolution", ha="center", va="center", fontsize=8)
            ax.set_axis_off()
            continue

        is_best = (best_idx is not None and idx == best_idx)
        title = f"Run {idx + 1} (cost={cost:.1f})" + (" [BEST]" if is_best else "")
        plot_route_on_axes(ax, instance, route, sorties, title, highlight_best=is_best)

    for idx in range(n_runs, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].set_axis_off()

    fig.suptitle(f"All routes for {instance_name} (best highlighted)", fontsize=14)
    fig.tight_layout()

    _show_and_save(fig, save_path)


def plot_run_costs_bars(
    instance_name: str,
    costs: List[float],
    run_numbers: List[int],
    save_path: Optional[Path] = None
) -> None:
    """
    Bar chart: cost per run.
    """
    x_pos = list(range(len(run_numbers)))

    fig = _get_single_figure((8, 4))
    ax = fig.add_subplot(111)

    ax.bar(x_pos, costs)
    ax.set_ylabel("Cost")
    ax.set_title(f"Costs per run - {instance_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Run index")
    fig.tight_layout()

    _show_and_save(fig, save_path)


def plot_run_times_bars(
    instance_name: str,
    times: List[float],
    run_numbers: List[int],
    save_path: Optional[Path] = None
) -> None:
    """
    Bar chart: runtime per run.
    """
    x_pos = list(range(len(run_numbers)))

    fig = _get_single_figure((8, 4))
    ax = fig.add_subplot(111)

    ax.bar(x_pos, times)
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtimes per run - {instance_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Run index")
    fig.tight_layout()

    _show_and_save(fig, save_path)


def plot_run_expansions_bars(
    instance_name: str,
    expansions: List[int],
    run_numbers: List[int],
    save_path: Optional[Path] = None
) -> None:
    """
    Bar chart: expansions per run.
    """
    x_pos = list(range(len(run_numbers)))

    fig = _get_single_figure((8, 4))
    ax = fig.add_subplot(111)

    ax.bar(x_pos, expansions)
    ax.set_ylabel("Expansions")
    ax.set_title(f"Expansions per run - {instance_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Run index")
    fig.tight_layout()

    _show_and_save(fig, save_path)


def plot_run_customers_bars(
    instance_name: str,
    truck_customers: List[int],
    drone_customers: List[int],
    run_numbers: List[int],
    save_path: Optional[Path] = None
) -> None:
    """
    Bar chart: customers served by truck vs drone per run.
    """
    if not run_numbers:
        return

    x_pos = list(range(len(run_numbers)))
    width = 0.35

    fig = _get_single_figure((8, 4))
    ax = fig.add_subplot(111)

    ax.bar([x - width / 2 for x in x_pos], truck_customers, width=width, label="Truck customers")
    ax.bar([x + width / 2 for x in x_pos], drone_customers, width=width, label="Drone customers")

    ax.set_ylabel("Number of customers")
    ax.set_title(f"Customers served per run - {instance_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Run index")
    ax.legend()
    fig.tight_layout()

    _show_and_save(fig, save_path)


def plot_gaps_per_instance(
    instance_keys: List[str],
    gaps_best: List[float],
    gaps_avg: List[float],
    save_path: Optional[Path] = None
) -> None:
    """
    Bar chart: GAP(best) and GAP(avg) vs optimal.
    """
    if not instance_keys:
        return

    x_pos = list(range(len(instance_keys)))
    width = 0.35

    fig = _get_single_figure((10, 5))
    ax = fig.add_subplot(111)

    ax.bar([x - width / 2 for x in x_pos], gaps_best, width=width, label="GAP(best)")
    ax.bar([x + width / 2 for x in x_pos], gaps_avg, width=width, label="GAP(avg)")
    ax.axhline(0.0, color="black", linewidth=1.0)

    ax.set_ylabel("GAP (%)")
    ax.set_title("GAP vs Gurobi optimal per instance")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(instance_keys, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    _show_and_save(fig, save_path)


def plot_ratio_best_per_instance(
    instance_keys: List[str],
    ratios_best: List[float],
    save_path: Optional[Path] = None
) -> None:
    """
    Bar chart: best_cost / optimal_cost per instance.
    """
    if not instance_keys:
        return

    x_pos = list(range(len(instance_keys)))

    fig = _get_single_figure((10, 5))
    ax = fig.add_subplot(111)

    ax.bar(x_pos, ratios_best)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Optimal ratio = 1.0")

    ax.set_ylabel("best_cost / optimal_cost")
    ax.set_title("Best-to-optimal cost ratio per instance")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(instance_keys, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    _show_and_save(fig, save_path)


# ============================================================
# Tabu Search
# ============================================================

def tabu_search(
    instance: FSTSPInstance,
    max_iters: int = MAX_ITERS,
    tabu_tenure: int = TABU_TENURE,
    max_neighbors: int = MAX_NEIGHBORS,
    time_limit_seconds: float = TIME_LIMIT_TS_RUN
) -> Tuple[float, List[int], List[int], List[DroneSortie], int]:
    """
    Tabu Search on permutations (swap neighborhood).
    Returns:
      final_cost, best_perm, best_truck_route, best_sorties, expansions
    """
    start_time = time.time()
    expansions = 0

    # Initial solution
    current_perm = build_initial_perm(instance)
    current_cost, _, _ = evaluate_perm(instance, current_perm)
    expansions += 1

    # Random restarts if infeasible
    tries = 0
    while math.isinf(current_cost) and tries < 20:
        random.shuffle(current_perm)
        current_cost, _, _ = evaluate_perm(instance, current_perm)
        expansions += 1
        tries += 1

    if math.isinf(current_cost):
        return math.inf, current_perm, [], [], expansions

    best_perm = current_perm[:]
    best_cost = current_cost

    # Tabu list: move signature -> expiry iteration
    tabu: Dict[Tuple[int, int], int] = {}
    iteration = 0

    while iteration < max_iters and (time.time() - start_time) < time_limit_seconds:
        iteration += 1
        best_candidate_perm: Optional[List[int]] = None
        best_candidate_cost = math.inf
        best_candidate_move: Optional[Tuple[int, int]] = None

        n = len(current_perm)
        if n < 2:
            break

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

            neighbor_perm = current_perm[:]
            neighbor_perm[i], neighbor_perm[j] = neighbor_perm[j], neighbor_perm[i]

            cost, _, _ = evaluate_perm(instance, neighbor_perm)
            expansions += 1

            if math.isinf(cost):
                continue

            is_tabu = move_sig in tabu and tabu[move_sig] > iteration
            if is_tabu and cost >= best_cost:
                continue

            if cost < best_candidate_cost:
                best_candidate_cost = cost
                best_candidate_perm = neighbor_perm
                best_candidate_move = move_sig

        # If no feasible neighbor, restart from heuristic
        if best_candidate_perm is None:
            current_perm = build_initial_perm(instance)
            current_cost, _, _ = evaluate_perm(instance, current_perm)
            expansions += 1
            if math.isinf(current_cost):
                break
            continue

        current_perm = best_candidate_perm
        current_cost = best_candidate_cost

        if best_candidate_move is not None:
            tabu[best_candidate_move] = iteration + tabu_tenure

        if iteration % 50 == 0:
            expired = [k for k, exp in tabu.items() if exp <= iteration]
            for k in expired:
                del tabu[k]

        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = current_perm[:]

    final_cost, best_truck_route, best_sorties = evaluate_perm(instance, best_perm)
    expansions += 1
    return final_cost, best_perm, best_truck_route, best_sorties, expansions


# ============================================================
# Run all instances + plots + report folder
# ============================================================

def run_all_instances(n_runs_per_instance: int = N_RUNS_PER_INSTANCE) -> None:
    """
    For each instance:
      - run Tabu Search multiple times
      - compute statistics (finite runs only)
      - save plots per instance
    At the end:
      - save global plots into results/report
      - save a summary CSV into results/report
      - print total execution time
      - close plot window
    """
    plt.ion()  # interactive mode for single-window updating

    # Total execution timer
    total_start_time = time.time()

    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path().resolve()

    instances_dir = base_dir / "instances"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report_dir = results_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(instances_dir.glob("FSTSP-*.txt"))
    if not txt_files:
        print(f"No instances found in {instances_dir}")
        plt.ioff()
        plt.close("all")
        return

    # Global lists for report plots
    instances_with_opt: List[str] = []
    gaps_best_all: List[float] = []
    gaps_avg_all: List[float] = []
    ratios_best_all: List[float] = []

    # Summary table rows
    summary_rows: List[Dict[str, object]] = []

    for path in txt_files:
        instance_name = path.name
        instance_key = path.stem

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
            random.seed(run_idx)

            t0_run = time.time()
            cost, perm, truck_route, sorties, expansions = tabu_search(instance)
            runtime_run = time.time() - t0_run

            all_costs.append(cost)
            run_times.append(runtime_run)
            run_expansions.append(expansions)
            run_truck_routes.append(truck_route)
            run_sorties_list.append(sorties)

            print(f"  Run {run_idx:2d}: cost = {cost:.2f}, time = {runtime_run:.2f} s, expansions = {expansions}")

            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_perm = perm[:]
                best_overall_truck_route = truck_route[:]
                best_overall_sorties = list(sorties)

        inst_runtime = time.time() - t0_inst

        print(f"\nInstance {instance_name} summary:")
        if all(math.isinf(c) for c in all_costs):
            print("  No feasible solution found in any run.")
            print(f"  Total time for {n_runs_per_instance} runs: {inst_runtime:.2f} seconds\n")
            continue

        # Finite runs only
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

        # Customers served by truck/drone per run (finite runs)
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
            truck_customers_run = [c for c in route[1:-1] if c not in drone_customers_run]

            truck_customers_counts.append(len(truck_customers_run))
            drone_customers_counts.append(len(drone_customers_run))

        avg_truck_customers = statistics.mean(truck_customers_counts) if truck_customers_counts else 0.0
        avg_drone_customers = statistics.mean(drone_customers_counts) if drone_customers_counts else 0.0

        print(f"  Best cost over runs: {best_cost:.2f}")
        print(f"  Average cost (finite runs): {avg_cost:.2f}")
        print(f"  Std. dev. of cost: {std_cost:.2f}")
        print(f"  Average runtime per run (finite): {avg_time:.2f} s")
        print(f"  Std. dev. runtime: {std_time:.2f} s")
        print(f"  Average expansions per run (finite): {avg_exp:.1f}")
        print(f"  Std. dev. expansions: {std_exp:.1f}")
        print(f"  Avg. customers served by truck: {avg_truck_customers:.2f}")
        print(f"  Avg. customers served by drone: {avg_drone_customers:.2f}")
        print(f"  Total time for {n_runs_per_instance} runs: {inst_runtime:.2f} seconds")

        # Gaps vs optimal (if available)
        opt_val = OPTIMAL_OBJECTIVES.get(instance_key)
        gap_best = ""
        gap_avg = ""
        ratio_best = ""

        if opt_val is not None:
            gap_best_val = 100.0 * (best_cost - opt_val) / opt_val
            gap_avg_val = 100.0 * (avg_cost - opt_val) / opt_val
            ratio_best_val = best_cost / opt_val

            gap_best = f"{gap_best_val:.6f}"
            gap_avg = f"{gap_avg_val:.6f}"
            ratio_best = f"{ratio_best_val:.6f}"

            print(f"  Gurobi optimal objective: {opt_val:.2f}")
            print(f"  GAP(best)  = {gap_best_val:.2f}%")
            print(f"  GAP(avg)   = {gap_avg_val:.2f}%")

            instances_with_opt.append(instance_key)
            gaps_best_all.append(gap_best_val)
            gaps_avg_all.append(gap_avg_val)
            ratios_best_all.append(ratio_best_val)
        else:
            print("  [INFO] No Gurobi optimal value stored for this instance; gaps not computed.")

        # Add row to summary table
        summary_rows.append({
            "instance": instance_key,
            "opt_value": "" if opt_val is None else f"{opt_val:.6f}",
            "best_cost": f"{best_cost:.6f}",
            "avg_cost": f"{avg_cost:.6f}",
            "std_cost": f"{std_cost:.6f}",
            "gap_best": gap_best,
            "gap_avg": gap_avg,
            "ratio_best": ratio_best,
            "avg_time": f"{avg_time:.6f}",
            "std_time": f"{std_time:.6f}",
            "avg_exp": f"{avg_exp:.6f}",
            "std_exp": f"{std_exp:.6f}",
            "avg_truck_customers": f"{avg_truck_customers:.6f}",
            "avg_drone_customers": f"{avg_drone_customers:.6f}",
        })

        # Plots (per instance)
        if best_overall_perm is not None:
            plot_run_costs_bars(
                instance_name,
                finite_costs,
                finite_run_numbers,
                save_path=instance_results_dir / "costs_per_run.png",
            )
            plot_run_times_bars(
                instance_name,
                finite_times,
                finite_run_numbers,
                save_path=instance_results_dir / "times_per_run.png",
            )
            plot_run_expansions_bars(
                instance_name,
                finite_expansions,
                finite_run_numbers,
                save_path=instance_results_dir / "expansions_per_run.png",
            )
            plot_run_customers_bars(
                instance_name,
                truck_customers_counts,
                drone_customers_counts,
                finite_run_numbers,
                save_path=instance_results_dir / "customers_per_run.png",
            )
            plot_all_routes_subplots(
                instance_name,
                instance,
                run_truck_routes,
                run_sorties_list,
                all_costs,
                save_path=instance_results_dir / "routes_all_runs.png",
            )

        print("\n" + "=" * 60 + "\n")

    # Global plots for the report
    if instances_with_opt:
        plot_gaps_per_instance(
            instances_with_opt,
            gaps_best_all,
            gaps_avg_all,
            save_path=report_dir / "gaps_vs_gurobi.png",
        )
        plot_ratio_best_per_instance(
            instances_with_opt,
            ratios_best_all,
            save_path=report_dir / "ratio_vs_gurobi.png",
        )

    # Summary CSV for the report
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

        with table_path.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row[h]) for h in header) + "\n")

        print(f"Summary table for report saved to: {table_path}")

    # Total execution time
    total_elapsed = time.time() - total_start_time
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)
    print(f"\nTOTAL EXECUTION TIME: {total_minutes} min {total_seconds} s\n")

    # Close plot window at the end
    plt.ioff()
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    run_all_instances()
