import os
import time
import math
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Global configuration
# ============================================================

SEED = 420
np.random.seed(SEED)
random.seed(SEED)

RESULT_DIR = "results_project2_approach21_22"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "tables"), exist_ok=True)


# ============================================================
# Data structures
# ============================================================

@dataclass
class InstanceConfig:
    n: int = 10000
    m: int = 10
    inventory_level: int = 1000
    bernoulli_p: float = 0.3
    noise_std: float = 0.2
    price_low: float = 1.0
    price_high: float = 2.8
    case_name: str = "case_standard"


@dataclass
class MethodConfig:
    method_name: str
    utility_type: str  # "log" or "exp"
    w: float
    a: Optional[float] = None  # used only for exp utility
    k0: int = 100
    update_points: Optional[List[int]] = None  # used for dynamic 2.2
    solver: str = "ECOS"
    eps_log: float = 1e-6


@dataclass
class RunResult:
    case_name: str
    seed: int
    method_name: str
    utility_type: str
    w: float
    a: Optional[float]
    k0: int
    revenue: float
    offline_revenue: float
    revenue_ratio: float
    runtime_sec: float
    accept_rate: float
    accepted_count: int
    n: int
    m: int
    inventory_total: float
    inventory_used_total: float
    inventory_utilization: float
    resource_used: List[float]
    resource_left: List[float]
    update_points: Optional[List[int]] = None


# ============================================================
# Utility functions
# ============================================================

def set_global_seed(seed: int = 420) -> None:
    np.random.seed(seed)
    random.seed(seed)


def solve_problem(problem: cp.Problem, preferred_solver: str = "ECOS") -> None:
    """
    Try preferred solver first, then fallback to SCS.
    """
    try:
        problem.solve(solver=preferred_solver, verbose=False)
        if problem.status in ["optimal", "optimal_inaccurate"]:
            return
    except Exception:
        pass

    try:
        problem.solve(solver="SCS", verbose=False)
    except Exception as e:
        raise RuntimeError(f"Both preferred solver {preferred_solver} and SCS failed: {e}")

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Data generation
# ============================================================

def generate_instance(cfg: InstanceConfig, seed: int = 420) -> Dict[str, np.ndarray]:
    """
    Generate one auction-market instance according to the project description.

    a_k: binary resource request vector
    pi_k = p_true^T a_k + Gaussian noise
    """
    rng = np.random.default_rng(seed)

    n, m = cfg.n, cfg.m

    p_true = np.linspace(cfg.price_low, cfg.price_high, m)
    A = rng.binomial(1, cfg.bernoulli_p, size=(n, m)).astype(float)

    # Avoid completely empty requests
    empty_rows = np.where(A.sum(axis=1) == 0)[0]
    for idx in empty_rows:
        chosen = rng.integers(0, m)
        A[idx, chosen] = 1.0

    noise = rng.normal(loc=0.0, scale=cfg.noise_std, size=n)
    pi = A @ p_true + noise

    b = np.full(m, cfg.inventory_level, dtype=float)

    return {
        "A": A,
        "pi": pi,
        "b": b,
        "p_true": p_true,
    }


def build_case_configs() -> List[InstanceConfig]:
    return [
        InstanceConfig(
            n=10000,
            m=10,
            inventory_level=1000,
            bernoulli_p=0.3,
            noise_std=0.2,
            case_name="case_standard",
        ),
        InstanceConfig(
            n=10000,
            m=10,
            inventory_level=1000,
            bernoulli_p=0.3,
            noise_std=0.2,
            case_name="case_scarce",
        ),
        InstanceConfig(
            n=10000,
            m=10,
            inventory_level=1000,
            bernoulli_p=0.3,
            noise_std=0.5,
            case_name="case_high_noise",
        ),
    ]


def postprocess_case(case_cfg: InstanceConfig, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Modify inventory structure for different cases.
    """
    if case_cfg.case_name == "case_scarce":
        b = data["b"].copy()
        b[:3] = 500.0
        data["b"] = b
    return data


# ============================================================
# Offline benchmark
# ============================================================

def solve_offline_lp(A: np.ndarray, pi: np.ndarray, b: np.ndarray, solver: str = "ECOS") -> Dict:
    """
    Offline LP benchmark:
        max sum_j pi_j x_j
        s.t. A^T x <= b
             0 <= x <= 1
    """
    n = A.shape[0]

    x = cp.Variable(n)
    constraints = [
        A.T @ x <= b,
        x >= 0,
        x <= 1,
    ]
    objective = cp.Maximize(pi @ x)
    problem = cp.Problem(objective, constraints)
    solve_problem(problem, preferred_solver=solver)

    x_val = np.array(x.value).reshape(-1)
    resource_used = A.T @ x_val
    resource_left = b - resource_used

    return {
        "objective": float(problem.value),
        "x": x_val,
        "resource_used": resource_used,
        "resource_left": resource_left,
    }


# ============================================================
# Partial convex program for Approach 2.1 / 2.2
# ============================================================

def solve_partial_scpm(
    A_k: np.ndarray,
    pi_k: np.ndarray,
    b_scaled: np.ndarray,
    utility_type: str,
    w: float,
    a_param: Optional[float] = None,
    eps_log: float = 1e-6,
    solver: str = "ECOS",
) -> Dict:
    """
    Solve the revealed partial convex program (SCPM):

        max sum_{j=1}^k pi_j x_j + u(s)
        s.t. A_k^T x + s <= b_scaled
             0 <= x <= 1
             s >= 0

    We use <= instead of equality for practical numerical robustness.
    Dual prices are taken from the resource constraints.
    """
    k, m = A_k.shape

    x = cp.Variable(k)
    s = cp.Variable(m)

    constraints = [
        A_k.T @ x + s <= b_scaled,
        x >= 0,
        x <= 1,
        s >= 0,
    ]

    if utility_type == "log":
        # s must be strictly positive numerically
        constraints.append(s >= eps_log)
        utility = w * cp.sum(cp.log(s))
    elif utility_type == "exp":
        if a_param is None:
            raise ValueError("a_param must be provided for exp utility.")
        utility = w * cp.sum(1 - cp.exp(-a_param * s))
    else:
        raise ValueError("utility_type must be either 'log' or 'exp'.")

    objective = cp.Maximize(pi_k @ x + utility)
    problem = cp.Problem(objective, constraints)
    solve_problem(problem, preferred_solver=solver)

    x_val = np.array(x.value).reshape(-1)
    s_val = np.array(s.value).reshape(-1)

    # Resource-price multipliers from the partial convex program
    # The first constraint is: A_k.T @ x + s <= b_scaled
    dual_prices = np.array(constraints[0].dual_value).reshape(-1)

    return {
        "objective": float(problem.value),
        "x": x_val,
        "s": s_val,
        "dual_prices": dual_prices,
    }


# ============================================================
# Online execution rules
# ============================================================

def accept_bid(
    a_j: np.ndarray,
    pi_j: float,
    dual_prices: np.ndarray,
    remaining_inventory: np.ndarray,
) -> int:
    """
    Online rule:
        x_j = 1 if pi_j > a_j^T y and enough inventory remains
        x_j = 0 otherwise
    """
    if np.any(a_j > remaining_inventory + 1e-12):
        return 0

    threshold = float(a_j @ dual_prices)
    return 1 if pi_j > threshold else 0


# ============================================================
# Approach 2.1: One-time inventory-aware learning
# ============================================================

def run_approach_21(
    A: np.ndarray,
    pi: np.ndarray,
    b: np.ndarray,
    method_cfg: MethodConfig,
    offline_revenue: float,
) -> RunResult:
    """
    Approach 2.1:
    - Observe first k0 bidders
    - Solve SCPM using scaled inventory (k0/n) * b
    - Use the learned dual prices for all future bidders
    """
    start_time = time.time()

    n, m = A.shape
    k0 = method_cfg.k0

    A_k = A[:k0]
    pi_k = pi[:k0]
    b_scaled = (k0 / n) * b

    learned = solve_partial_scpm(
        A_k=A_k,
        pi_k=pi_k,
        b_scaled=b_scaled,
        utility_type=method_cfg.utility_type,
        w=method_cfg.w,
        a_param=method_cfg.a,
        eps_log=method_cfg.eps_log,
        solver=method_cfg.solver,
    )

    dual_prices = learned["dual_prices"]

    remaining = b.copy()
    revenue = 0.0
    accepted_count = 0

    # First k0 bidders are used only for learning
    for j in range(k0, n):
        xj = accept_bid(A[j], pi[j], dual_prices, remaining)
        if xj == 1:
            remaining -= A[j]
            revenue += float(pi[j])
            accepted_count += 1

    resource_used = b - remaining
    runtime_sec = time.time() - start_time

    return RunResult(
        case_name="",
        seed=SEED,
        method_name=method_cfg.method_name,
        utility_type=method_cfg.utility_type,
        w=method_cfg.w,
        a=method_cfg.a,
        k0=k0,
        revenue=revenue,
        offline_revenue=offline_revenue,
        revenue_ratio=revenue / offline_revenue if offline_revenue > 0 else np.nan,
        runtime_sec=runtime_sec,
        accept_rate=accepted_count / max(1, n - k0),
        accepted_count=accepted_count,
        n=n,
        m=m,
        inventory_total=float(b.sum()),
        inventory_used_total=float(resource_used.sum()),
        inventory_utilization=float(resource_used.sum() / max(1e-12, b.sum())),
        resource_used=resource_used.tolist(),
        resource_left=remaining.tolist(),
        update_points=None,
    )


# ============================================================
# Approach 2.2: Dynamic inventory-aware learning
# ============================================================

def default_update_points(n: int) -> List[int]:
    points = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    return [k for k in points if k < n]


def run_approach_22(
    A: np.ndarray,
    pi: np.ndarray,
    b: np.ndarray,
    method_cfg: MethodConfig,
    offline_revenue: float,
) -> RunResult:
    """
    Approach 2.2:
    - Recompute SCPM prices at update points
    - Use the newly computed prices for the immediate subsequent period
    """
    start_time = time.time()

    n, m = A.shape
    update_points = method_cfg.update_points if method_cfg.update_points is not None else default_update_points(n)
    update_points = sorted(set([k for k in update_points if 1 <= k < n]))

    remaining = b.copy()
    revenue = 0.0
    accepted_count = 0

    # Learned prices at each update point
    dual_prices = None
    current_update_idx = 0

    # We use bidders before the first update point only for learning
    first_k = update_points[0]

    for j in range(n):
        # Recompute prices exactly at update points
        if current_update_idx < len(update_points) and j == update_points[current_update_idx]:
            k = update_points[current_update_idx]
            A_k = A[:k]
            pi_k = pi[:k]
            b_scaled = (k / n) * b

            learned = solve_partial_scpm(
                A_k=A_k,
                pi_k=pi_k,
                b_scaled=b_scaled,
                utility_type=method_cfg.utility_type,
                w=method_cfg.w,
                a_param=method_cfg.a,
                eps_log=method_cfg.eps_log,
                solver=method_cfg.solver,
            )
            dual_prices = learned["dual_prices"]
            current_update_idx += 1
            continue

        # Before first learning point, do nothing
        if j < first_k:
            continue

        if dual_prices is None:
            continue

        xj = accept_bid(A[j], pi[j], dual_prices, remaining)
        if xj == 1:
            remaining -= A[j]
            revenue += float(pi[j])
            accepted_count += 1

    resource_used = b - remaining
    runtime_sec = time.time() - start_time

    effective_online_count = n - first_k

    return RunResult(
        case_name="",
        seed=SEED,
        method_name=method_cfg.method_name,
        utility_type=method_cfg.utility_type,
        w=method_cfg.w,
        a=method_cfg.a,
        k0=first_k,
        revenue=revenue,
        offline_revenue=offline_revenue,
        revenue_ratio=revenue / offline_revenue if offline_revenue > 0 else np.nan,
        runtime_sec=runtime_sec,
        accept_rate=accepted_count / max(1, effective_online_count),
        accepted_count=accepted_count,
        n=n,
        m=m,
        inventory_total=float(b.sum()),
        inventory_used_total=float(resource_used.sum()),
        inventory_utilization=float(resource_used.sum() / max(1e-12, b.sum())),
        resource_used=resource_used.tolist(),
        resource_left=remaining.tolist(),
        update_points=update_points,
    )


# ============================================================
# Experiment runner
# ============================================================

def build_method_grid() -> List[MethodConfig]:
    methods = []

    # Approach 2.1 - log utility
    for w in [1.0, 5.0, 10.0, 20.0]:
        methods.append(MethodConfig(
            method_name="Approach2.1",
            utility_type="log",
            w=w,
            a=None,
            k0=100,
            solver="ECOS",
        ))

    # Approach 2.1 - exp utility
    for w in [1.0, 5.0, 10.0]:
        for a in [0.01, 0.05, 0.1]:
            methods.append(MethodConfig(
                method_name="Approach2.1",
                utility_type="exp",
                w=w,
                a=a,
                k0=100,
                solver="ECOS",
            ))

    # Approach 2.2 - log utility
    for w in [1.0, 5.0, 10.0, 20.0]:
        methods.append(MethodConfig(
            method_name="Approach2.2",
            utility_type="log",
            w=w,
            a=None,
            k0=50,
            update_points=[50, 100, 200, 400, 800, 1600, 3200, 6400],
            solver="ECOS",
        ))

    # Approach 2.2 - exp utility
    for w in [1.0, 5.0, 10.0]:
        for a in [0.01, 0.05, 0.1]:
            methods.append(MethodConfig(
                method_name="Approach2.2",
                utility_type="exp",
                w=w,
                a=a,
                k0=50,
                update_points=[50, 100, 200, 400, 800, 1600, 3200, 6400],
                solver="ECOS",
            ))

    return methods


def run_single_case(case_cfg: InstanceConfig, seed: int = 420) -> pd.DataFrame:
    print(f"Running {case_cfg.case_name} ...")
    data = generate_instance(case_cfg, seed=seed)
    data = postprocess_case(case_cfg, data)

    A = data["A"]
    pi = data["pi"]
    b = data["b"]

    offline = solve_offline_lp(A, pi, b, solver="ECOS")
    offline_revenue = offline["objective"]

    save_json(
        {
            "case_name": case_cfg.case_name,
            "seed": seed,
            "offline_revenue": offline_revenue,
            "offline_resource_used": offline["resource_used"].tolist(),
            "offline_resource_left": offline["resource_left"].tolist(),
        },
        os.path.join(RESULT_DIR, "tables", f"{case_cfg.case_name}_offline.json"),
    )

    rows = []
    method_grid = build_method_grid()

    for method_cfg in method_grid:
        print(f"  -> {method_cfg.method_name}, utility={method_cfg.utility_type}, w={method_cfg.w}, a={method_cfg.a}")
        try:
            if method_cfg.method_name == "Approach2.1":
                result = run_approach_21(A, pi, b, method_cfg, offline_revenue)
            elif method_cfg.method_name == "Approach2.2":
                result = run_approach_22(A, pi, b, method_cfg, offline_revenue)
            else:
                raise ValueError(f"Unknown method: {method_cfg.method_name}")

            result.case_name = case_cfg.case_name
            rows.append(asdict(result))

        except Exception as e:
            print(f"     Failed: {e}")
            rows.append({
                "case_name": case_cfg.case_name,
                "seed": seed,
                "method_name": method_cfg.method_name,
                "utility_type": method_cfg.utility_type,
                "w": method_cfg.w,
                "a": method_cfg.a,
                "k0": method_cfg.k0,
                "revenue": np.nan,
                "offline_revenue": offline_revenue,
                "revenue_ratio": np.nan,
                "runtime_sec": np.nan,
                "accept_rate": np.nan,
                "accepted_count": np.nan,
                "n": A.shape[0],
                "m": A.shape[1],
                "inventory_total": float(b.sum()),
                "inventory_used_total": np.nan,
                "inventory_utilization": np.nan,
                "resource_used": None,
                "resource_left": None,
                "update_points": method_cfg.update_points,
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULT_DIR, "tables", f"{case_cfg.case_name}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return df


# ============================================================
# Plotting
# ============================================================

def plot_revenue_ratios(df: pd.DataFrame, case_name: str) -> None:
    plot_df = df.dropna(subset=["revenue_ratio"]).copy()
    if plot_df.empty:
        return

    plot_df["label"] = (
        plot_df["method_name"] + "-" +
        plot_df["utility_type"] + "-w=" +
        plot_df["w"].astype(str) +
        plot_df["a"].apply(lambda x: f"-a={x}" if pd.notna(x) else "")
    )

    plot_df = plot_df.sort_values(by="revenue_ratio", ascending=False).head(12)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["label"], plot_df["revenue_ratio"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Online Revenue / Offline Revenue")
    plt.title(f"Top Revenue Ratios - {case_name}")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "figures", f"{case_name}_top_revenue_ratios.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_runtime_vs_method(df: pd.DataFrame, case_name: str) -> None:
    plot_df = df.dropna(subset=["runtime_sec"]).copy()
    if plot_df.empty:
        return

    agg = (
        plot_df.groupby(["method_name", "utility_type"])["runtime_sec"]
        .mean()
        .reset_index()
    )
    agg["label"] = agg["method_name"] + "-" + agg["utility_type"]

    plt.figure(figsize=(8, 5))
    plt.bar(agg["label"], agg["runtime_sec"])
    plt.ylabel("Runtime (sec)")
    plt.title(f"Average Runtime by Method/Utility - {case_name}")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "figures", f"{case_name}_runtime.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_inventory_utilization(df: pd.DataFrame, case_name: str) -> None:
    plot_df = df.dropna(subset=["inventory_utilization"]).copy()
    if plot_df.empty:
        return

    plot_df["label"] = (
        plot_df["method_name"] + "-" +
        plot_df["utility_type"] + "-w=" +
        plot_df["w"].astype(str)
    )
    plot_df = plot_df.sort_values(by="inventory_utilization", ascending=False).head(12)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["label"], plot_df["inventory_utilization"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Inventory Utilization")
    plt.title(f"Top Inventory Utilization - {case_name}")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "figures", f"{case_name}_inventory_utilization.png")
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    set_global_seed(SEED)

    all_results = []

    for case_cfg in build_case_configs():
        df_case = run_single_case(case_cfg, seed=SEED)
        all_results.append(df_case)

        plot_revenue_ratios(df_case, case_cfg.case_name)
        plot_runtime_vs_method(df_case, case_cfg.case_name)
        plot_inventory_utilization(df_case, case_cfg.case_name)

    df_all = pd.concat(all_results, axis=0, ignore_index=True)
    all_csv = os.path.join(RESULT_DIR, "tables", "all_cases_results.csv")
    df_all.to_csv(all_csv, index=False)

    print("\nFinished all experiments.")
    print(f"Master result table saved to: {all_csv}")


if __name__ == "__main__":
    main()