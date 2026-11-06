import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from optimization.generators import generate_problem_data
from optimization.losses import make_quadratic_loss
from optimization.runner import run_optimization

N_VALUES = list(range(100, 1001, 100))          # 100, 200, ..., 1000
KAPPA_VALUES = list(range(1, 11))               # 1, 2, ..., 10

NUM_ITERS = 3000
TOL_DIST = 0.05

LR = 1e-2
BETAS_ADAM = (0.9, 0.999)
MOMENTUM_NESTEROV = 0.9

SEED_A = 42
SEED_B = 43
SEED_U0 = 44

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)



def run_experiment():
    """Перебирает (n, kappa), запускает Adam и Nesterov."""
    optimizers = {
        "Adam":     (torch.optim.Adam, dict(lr=LR, betas=BETAS_ADAM)),
        # Nesterov = SGD с momentum и nesterov=True
        "Nesterov": (torch.optim.SGD, dict(lr=LR,
                                           momentum=MOMENTUM_NESTEROV,
                                           nesterov=True)),
    }

    results = {name: {n: {} for n in N_VALUES} for name in optimizers}

    for n in N_VALUES:
        print(f"\n=== n = {n} ===")
        for kappa in KAPPA_VALUES:
            data = generate_problem_data(
                n=n, kappa=float(kappa),
                seed_A=SEED_A, seed_b=SEED_B, seed_u0=SEED_U0,
                R=5.0,
            )
            loss_fn = make_quadratic_loss(data["A"], data["b"])

            for name, (opt_class, opt_kwargs) in optimizers.items():
                u_final, hist = run_optimization(
                    opt_class,
                    u0_np=data["u0"],
                    loss_fn=loss_fn,
                    u_star_np=data["u_star"],
                    num_iters=NUM_ITERS,
                    tol_dist=TOL_DIST,
                    **opt_kwargs,
                )
                results[name][n][kappa] = {
                    "time":       hist["time"],
                    "iters":      hist["iters"],
                    "dist_final": hist["dist"][-1] if hist["dist"] else float("nan"),
                }

            log = "  ".join(
                f"{name}: {results[name][n][kappa]['iters']}it/"
                f"{results[name][n][kappa]['time']:.3f}s"
                for name in optimizers
            )
            print(f"  kappa={kappa:2d}  {log}")

    return results

def aggregate_by_n(results):
    """
    Для каждого n: среднее по всем kappa
    """
    out = {}
    for name, by_n in results.items():
        time_mean, iters_mean = [], []
        for n in N_VALUES:
            times = [by_n[n][k]["time"]  for k in KAPPA_VALUES]
            iters = [by_n[n][k]["iters"] for k in KAPPA_VALUES]
            time_mean.append(np.mean(times))
            iters_mean.append(np.mean(iters))
        out[name] = {
            "n": N_VALUES,
            "time_mean": time_mean,
            "iters_mean": iters_mean,
        }
    return out


def aggregate_by_kappa(results):
    """
    Для каждого kappa: среднее по всем n
    """
    out = {}
    for name, by_n in results.items():
        time_mean, iters_mean = [], []
        for kappa in KAPPA_VALUES:
            times = [by_n[n][kappa]["time"]  for n in N_VALUES]
            iters = [by_n[n][kappa]["iters"] for n in N_VALUES]
            time_mean.append(np.mean(times))
            iters_mean.append(np.mean(iters))
        out[name] = {
            "kappa": KAPPA_VALUES,
            "time_mean": time_mean,
            "iters_mean": iters_mean,
        }
    return out


#визуализация

def plot_all(results):
    data_n = aggregate_by_n(results) # усреднено по kappa
    data_k = aggregate_by_kappa(results) # усреднено по n

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in data_n.items():
        ax.plot(d["n"], d["time_mean"], marker="o", label=name)
    ax.set_xlabel("Размерность n")
    ax.set_ylabel("Время, с (среднее по kappa)")
    ax.set_title("Время от размерности (усреднено по kappa)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "time_vs_n.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in data_k.items():
        ax.plot(d["kappa"], d["time_mean"], marker="o", label=name)
    ax.set_xlabel("Число обусловленности kappa")
    ax.set_ylabel("Время, с (среднее по n)")
    ax.set_title("Время от обусловленности (усреднено по n)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "time_vs_kappa.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in data_n.items():
        ax.plot(d["n"], d["iters_mean"], marker="o", label=name)
    ax.set_xlabel("Размерность n")
    ax.set_ylabel("Итерации до сходимости (среднее по kappa)")
    ax.set_title("Итерации от размерности (усреднено по kappa)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "iters_vs_n.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in data_k.items():
        ax.plot(d["kappa"], d["iters_mean"], marker="o", label=name)
    ax.set_xlabel("Число обусловленности kappa")
    ax.set_ylabel("Итерации до сходимости (среднее по n)")
    ax.set_title("Итерации от обусловленности (усреднено по n)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "iters_vs_kappa.png", dpi=150)
    plt.close(fig)

    print(f"\nГрафики сохранены в {RESULTS_DIR.resolve()}")

if __name__ == "__main__":
    t0 = time.perf_counter()
    results = run_experiment()
    print(f"\nВсе эксперименты заняли {time.perf_counter() - t0:.1f} с")
    plot_all(results)