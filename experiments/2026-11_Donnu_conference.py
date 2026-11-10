import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from optimization.generators import generate_problem_data
from optimization.losses import make_quadratic_loss
from optimization.runner import run_optimization
from optimization.optimizers.collinear import CollinearGradients

N_VALUES = [10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
KAPPA_VALUES = [1, 2, 5, 10, 20, 50, 75, 100]

NUM_ITERS = 3000
TOL_DIST = 0.05

LR = 1e-2
BETAS_ADAM = (0.9, 0.999)
MOMENTUM_NESTEROV = 0.9

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MAX_ITERS_MAP = {
    "Adam": NUM_ITERS,
    "Метод Нестерова": NUM_ITERS,
    "Метод коллинеарных градиентов": 50,
}

OPTIMIZER_NAMES = ["Adam", "Метод Нестерова", "Метод коллинеарных градиентов"]


def seeds_for(n: int, kappa: int) -> tuple[int, int, int]:
    base = 1000 * n + kappa
    return base + 42, base + 1042, base + 2042


def make_optimizers(loss_fn):
    return {
        "Adam": (torch.optim.Adam, dict(lr=LR, betas=BETAS_ADAM)),
        # Nesterov = SGD с momentum и nesterov=True
        "Метод Нестерова": (torch.optim.SGD, dict(lr=LR,
                                                 momentum=MOMENTUM_NESTEROV,
                                                 nesterov=True)),
        "Метод коллинеарных градиентов": (CollinearGradients, dict(
            lr=1.0,
            objective_fn=loss_fn,
            c1=1e-8,
            c2=4.0,
            delta0=1e-5,
            delta_m_factor=1e-15,
            h0=1e-8,
        )),
    }


def run_experiment():
    """
    Перебирает (n, kappa), запускает Adam, Нестерова и CollGM.

    Возвращает:
        results     — {name: {n: {kappa: {time, iters, dist_final,
                                          f_evals, grad_evals, total_evals}}}}
        total_times — {name: суммарное время по всем (n, kappa)}
    """
    results = {name: {n: {} for n in N_VALUES} for name in OPTIMIZER_NAMES}
    total_times = {name: 0.0 for name in OPTIMIZER_NAMES}

    for n in N_VALUES:
        print(f"\n=== n = {n} ===")
        for kappa in KAPPA_VALUES:
            seed_A, seed_b, seed_u0 = seeds_for(n, kappa)
            data = generate_problem_data(
                n=n, kappa=float(kappa),
                seed_A=seed_A, seed_b=seed_b, seed_u0=seed_u0,
                R=5.0,
            )
            loss_fn = make_quadratic_loss(data["A"], data["b"])
            optimizers = make_optimizers(loss_fn)

            for name, (opt_class, opt_kwargs) in optimizers.items():
                u_final, hist = run_optimization(
                    opt_class,
                    u0_np=data["u0"],
                    loss_fn=loss_fn,
                    u_star_np=data["u_star"],
                    num_iters=MAX_ITERS_MAP[name],
                    tol_dist=TOL_DIST,
                    **opt_kwargs,
                )
                results[name][n][kappa] = {
                    "time":        hist["time"],
                    "iters":       hist["iters"],
                    "dist_final":  hist["dist"][-1] if hist["dist"] else float("nan"),
                    "f_evals":     hist["f_evals"],
                    "grad_evals":  hist["grad_evals"],
                    "total_evals": hist["f_evals"] + hist["grad_evals"],
                }
                total_times[name] += hist["time"]

            log = "  ".join(
                f"{name}: {results[name][n][kappa]['iters']}it/"
                f"{results[name][n][kappa]['time']:.3f}s/"
                f"{results[name][n][kappa]['total_evals']}ev"
                for name in optimizers
            )
            print(f"  kappa={kappa:3d}  {log}")

    return results, total_times


def aggregate_by_n(results):
    """Для каждого n: среднее по всем kappa (time, iters, total_evals)."""
    out = {}
    for name, by_n in results.items():
        time_mean, iters_mean, evals_mean = [], [], []
        for n in N_VALUES:
            time_mean.append(np.mean([by_n[n][k]["time"]        for k in KAPPA_VALUES]))
            iters_mean.append(np.mean([by_n[n][k]["iters"]      for k in KAPPA_VALUES]))
            evals_mean.append(np.mean([by_n[n][k]["total_evals"] for k in KAPPA_VALUES]))
        out[name] = {
            "n": N_VALUES,
            "time_mean": time_mean,
            "iters_mean": iters_mean,
            "evals_mean": evals_mean,
        }
    return out


def aggregate_by_kappa(results):
    """Для каждого kappa: среднее по всем n (time, iters, total_evals)."""
    out = {}
    for name, by_n in results.items():
        time_mean, iters_mean, evals_mean = [], [], []
        for kappa in KAPPA_VALUES:
            time_mean.append(np.mean([by_n[n][kappa]["time"]        for n in N_VALUES]))
            iters_mean.append(np.mean([by_n[n][kappa]["iters"]      for n in N_VALUES]))
            evals_mean.append(np.mean([by_n[n][kappa]["total_evals"] for n in N_VALUES]))
        out[name] = {
            "kappa": KAPPA_VALUES,
            "time_mean": time_mean,
            "iters_mean": iters_mean,
            "evals_mean": evals_mean,
        }
    return out

def plot_all(results):
    """
    Композитный рисунок 3x2
    """
    data_n = aggregate_by_n(results)
    data_k = aggregate_by_kappa(results)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    ax_a, ax_b = axes[0, 0], axes[0, 1] # время
    ax_c, ax_d = axes[1, 0], axes[1, 1] # итерации
    ax_e, ax_f = axes[2, 0], axes[2, 1] # вычисления

    # (a) время от n
    for name, d in data_n.items():
        ax_a.plot(d["n"], d["time_mean"], marker="o", label=name)
    ax_a.set_xlabel("Размерность n")
    ax_a.set_ylabel("Время, с")

    # (b) время от обусловленности
    for name, d in data_k.items():
        ax_b.plot(d["kappa"], d["time_mean"], marker="o", label=name)
    ax_b.set_xlabel("Число обусловленности $\\kappa$")
    ax_b.set_ylabel("Время, с")

    # (c) итерации от n
    for name, d in data_n.items():
        ax_c.plot(d["n"], d["iters_mean"], marker="o", label=name)
    ax_c.set_xlabel("Размерность n")
    ax_c.set_ylabel("Итерации")

    # (d) итерации от обусловленности
    for name, d in data_k.items():
        ax_d.plot(d["kappa"], d["iters_mean"], marker="o", label=name)
    ax_d.set_xlabel("Число обусловленности $\\kappa$")
    ax_d.set_ylabel("Итерации")

    # (e) количество вычислений (f + ∇f) от n
    for name, d in data_n.items():
        ax_e.plot(d["n"], d["evals_mean"], marker="o", label=name)
    ax_e.set_xlabel("Размерность n")
    ax_e.set_ylabel("Вычислений $f и \\nabla f$, шт.")
    # Если разброс большой — раскомментируй строку ниже (и удали plot выше):
    # ax_e.set_yscale("log")

    # (f) количество вычислений (f + ∇f) от обусловленности
    for name, d in data_k.items():
        ax_f.plot(d["kappa"], d["evals_mean"], marker="o", label=name)
    ax_f.set_xlabel("Число обусловленности $\\kappa$")
    ax_f.set_ylabel("Вычислений $f и \\nabla f$, шт.")
    # ax_f.set_yscale("log")

    # Общая легенда снизу
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    # Метки панелей
    for ax, label in [(ax_a, "(a)"), (ax_b, "(b)"),
                      (ax_c, "(c)"), (ax_d, "(d)"),
                      (ax_e, "(e)"), (ax_f, "(f)")]:
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold",
                va="bottom", ha="left")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = RESULTS_DIR / "comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nКомпозитный рисунок сохранён: {out_path.resolve()}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    results, total_times = run_experiment()
    print(f"\nВсе эксперименты заняли {time.perf_counter() - t0:.1f} с")

    print("\nСуммарное время решения всех задач по методам:")
    for name, t in total_times.items():
        print(f"  {name:35s} {t:10.2f} с")

    plot_all(results)