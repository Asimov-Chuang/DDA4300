"""Part 1 experiments for online resource allocation in auction markets.

This script implements the two algorithms in Approach 1:
    1. One-time learning with a fixed dual price.
    2. Dynamic updating with prices recomputed at doubling times.

Run from the project root:
    python part1/part1_experiment.py
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


@dataclass
class LPResult:
    value: float
    x: np.ndarray
    resource_price: np.ndarray
    elapsed: float


@dataclass
class OnlineResult:
    method: str
    revenue: float
    accepted: int
    usage: np.ndarray
    remaining: np.ndarray
    solve_time: float
    decision_time: float
    k: int | None = None
    update_times: np.ndarray | None = None
    price: np.ndarray | None = None
    price_path: np.ndarray | None = None


def generate_instance(
    n: int = 10_000,
    m: int = 10,
    inventory: float = 1_000.0,
    seed: int = 307,
    noise_sigma: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the online auction instance described in the project handout."""
    rng = np.random.default_rng(seed)
    capacities = inventory * np.ones(m)
    pbar = np.linspace(0.6, 1.5, m)
    requests = (rng.random((n, m)) < 0.5).astype(float)
    bids = requests @ pbar + noise_sigma * rng.standard_normal(n)
    return requests, bids, capacities, pbar


def solve_packing_lp(
    requests: np.ndarray,
    bids: np.ndarray,
    capacities: np.ndarray,
) -> LPResult:
    """Solve max bids^T x subject to requests.T @ x <= capacities, 0 <= x <= 1."""
    start = time.perf_counter()
    result = linprog(
        c=-bids,
        A_ub=requests.T,
        b_ub=capacities,
        bounds=(0.0, 1.0),
        method="highs",
    )
    elapsed = time.perf_counter() - start

    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")

    # HiGHS solves a minimization problem. Its inequality marginals are
    # minimization sensitivities, so we negate them to obtain max-LP prices.
    resource_price = -np.asarray(result.ineqlin.marginals)
    return LPResult(
        value=-float(result.fun),
        x=result.x,
        resource_price=resource_price,
        elapsed=elapsed,
    )


def run_one_time_learning(
    requests: np.ndarray,
    bids: np.ndarray,
    capacities: np.ndarray,
    k: int,
) -> OnlineResult:
    n = len(bids)
    learned = solve_packing_lp(requests[:k], bids[:k], (k / n) * capacities)

    accepted = np.zeros(n, dtype=bool)
    remaining = capacities.copy()

    start = time.perf_counter()
    for j in range(k, n):
        req = requests[j]
        threshold = req @ learned.resource_price
        if bids[j] > threshold and np.all(req <= remaining + 1e-10):
            accepted[j] = True
            remaining -= req
    decision_time = time.perf_counter() - start

    usage = capacities - remaining
    return OnlineResult(
        method="One-time",
        k=k,
        revenue=float(bids @ accepted.astype(float)),
        accepted=int(accepted.sum()),
        usage=usage,
        remaining=remaining,
        solve_time=learned.elapsed,
        decision_time=decision_time,
        price=learned.resource_price,
    )


def run_dynamic_updating(
    requests: np.ndarray,
    bids: np.ndarray,
    capacities: np.ndarray,
    update_times: np.ndarray,
) -> OnlineResult:
    n, m = requests.shape
    update_times = update_times[update_times < n]

    accepted = np.zeros(n, dtype=bool)
    remaining = capacities.copy()
    price_path = np.zeros((len(update_times), m))
    solve_time = 0.0
    decision_time = 0.0

    for idx, k in enumerate(update_times):
        learned = solve_packing_lp(requests[:k], bids[:k], (k / n) * capacities)
        price_path[idx] = learned.resource_price
        solve_time += learned.elapsed

        stop = int(update_times[idx + 1]) if idx + 1 < len(update_times) else n

        start = time.perf_counter()
        for j in range(int(k), stop):
            req = requests[j]
            threshold = req @ learned.resource_price
            if bids[j] > threshold and np.all(req <= remaining + 1e-10):
                accepted[j] = True
                remaining -= req
        decision_time += time.perf_counter() - start

    usage = capacities - remaining
    return OnlineResult(
        method="Dynamic",
        revenue=float(bids @ accepted.astype(float)),
        accepted=int(accepted.sum()),
        usage=usage,
        remaining=remaining,
        solve_time=solve_time,
        decision_time=decision_time,
        update_times=update_times,
        price_path=price_path,
    )


def summarize_rows(
    offline: LPResult,
    offline_usage: np.ndarray,
    capacities: np.ndarray,
    online_results: list[OnlineResult],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = [
        {
            "method": "Offline LP",
            "k": "",
            "revenue": offline.value,
            "competitive_ratio": 1.0,
            "accepted": int(np.sum(offline.x > 1e-8)),
            "avg_usage": float(np.mean(offline_usage / capacities)),
            "min_remaining": float(np.min(capacities - offline_usage)),
            "solve_time_sec": offline.elapsed,
            "decision_time_sec": 0.0,
        }
    ]

    for res in online_results:
        rows.append(
            {
                "method": res.method,
                "k": "" if res.k is None else res.k,
                "revenue": res.revenue,
                "competitive_ratio": res.revenue / offline.value,
                "accepted": res.accepted,
                "avg_usage": float(np.mean(res.usage / capacities)),
                "min_remaining": float(np.min(res.remaining)),
                "solve_time_sec": res.solve_time,
                "decision_time_sec": res.decision_time,
            }
        )

    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    header = (
        f"{'method':<12} {'k':>6} {'revenue':>12} {'ratio':>9} "
        f"{'accepted':>9} {'avg_usage':>10} {'solve_s':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:<12} {str(row['k']):>6} "
            f"{float(row['revenue']):>12.2f} "
            f"{float(row['competitive_ratio']):>9.4f} "
            f"{int(row['accepted']):>9} "
            f"{float(row['avg_usage']):>10.4f} "
            f"{float(row['solve_time_sec']):>9.3f}"
        )


def make_plots(
    k_set: np.ndarray,
    one_time_results: list[OnlineResult],
    dynamic_result: OnlineResult,
    offline: LPResult,
    offline_usage: np.ndarray,
    capacities: np.ndarray,
    result_dir: Path,
) -> None:
    ratios = np.array([res.revenue / offline.value for res in one_time_results])
    dynamic_ratio = dynamic_result.revenue / offline.value

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_set, ratios, marker="o", linewidth=1.8, label="One-time learning")
    ax.axhline(dynamic_ratio, color="tab:orange", linewidth=1.8, label="Dynamic updating")
    ax.set_xlabel("learning sample size k")
    ax.set_ylabel("competitive ratio")
    ax.set_title("Approach 1.1 and 1.2 revenue performance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "part1_revenue_ratio.png", dpi=200)
    plt.close(fig)

    best_one_time = max(one_time_results, key=lambda res: res.revenue)
    usage_matrix = np.column_stack(
        [
            offline_usage / capacities,
            best_one_time.usage / capacities,
            dynamic_result.usage / capacities,
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, len(capacities) + 1)
    width = 0.25
    ax.bar(x - width, usage_matrix[:, 0], width, label="Offline LP")
    ax.bar(x, usage_matrix[:, 1], width, label=f"One-time k={best_one_time.k}")
    ax.bar(x + width, usage_matrix[:, 2], width, label="Dynamic")
    ax.set_xlabel("resource index")
    ax.set_ylabel("used fraction of inventory")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_title("Resource utilization")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(result_dir / "part1_resource_usage.png", dpi=200)
    plt.close(fig)


def main() -> None:
    result_dir = Path(__file__).resolve().parent / "results"
    result_dir.mkdir(exist_ok=True)

    requests, bids, capacities, pbar = generate_instance()
    k_set = 50 * (2 ** np.arange(8))
    k_set = k_set[k_set < len(bids)]

    print("Solving offline LP benchmark...")
    offline = solve_packing_lp(requests, bids, capacities)
    offline_usage = requests.T @ offline.x

    print("Running one-time learning experiments...")
    one_time_results = [
        run_one_time_learning(requests, bids, capacities, int(k)) for k in k_set
    ]

    print("Running dynamic updating experiment...")
    dynamic_result = run_dynamic_updating(requests, bids, capacities, k_set.astype(int))

    rows = summarize_rows(
        offline=offline,
        offline_usage=offline_usage,
        capacities=capacities,
        online_results=[*one_time_results, dynamic_result],
    )
    print_table(rows)
    write_csv(rows, result_dir / "part1_results.csv")

    np.savez_compressed(
        result_dir / "part1_workspace.npz",
        requests=requests,
        bids=bids,
        capacities=capacities,
        pbar=pbar,
        k_set=k_set,
        offline_x=offline.x,
        offline_price=offline.resource_price,
        offline_usage=offline_usage,
        dynamic_price_path=dynamic_result.price_path,
    )

    make_plots(
        k_set=k_set,
        one_time_results=one_time_results,
        dynamic_result=dynamic_result,
        offline=offline,
        offline_usage=offline_usage,
        capacities=capacities,
        result_dir=result_dir,
    )
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
