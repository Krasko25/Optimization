import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from optimization.generators import generate_problem_data
from optimization.losses import make_quadratic_loss
from optimization.runner import run_optimization
from optimization.optimizers.collinear import CollinearGradients
from optimization.optimizers.collinear_old import CollinearGradients_old

N_VALUES = [10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
KAPPA_VALUES = [1, 2, 5, 10, 20, 50, 75, 100]

NUM_ITERS = 50
TOL_DIST = 0.05

# общие гиперпараметры для обеих версий МКГ —
# чтобы сравнение было честным (только реализация различается)
MKG_COMMON = dict(
    lr=1.0,
    c1=1e-8,
    c2=4.0,
    delta0=1e-5,
    delta_m_factor=1e-15,
    h0=1e-8,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MAX_ITERS_MAP = {
    "МКГ (старая)": NUM_ITERS,
    "МКГ (новая)":  NUM_ITERS,
}

OPTIMIZER_NAMES = [
    "МКГ (старая)",
    "МКГ (новая)",
]


def seeds_for(n: int, kappa: int) -> tuple[int, int, int]:
    base = 1000 * n + kappa
    return base + 42, base + 1042, base + 2042


def make_optimizers(loss_fn):
    return {
        "МКГ (старая)": (CollinearGradients_old, dict(
            objective_fn=loss_fn, **MKG_COMMON,
        )),
        "МКГ (новая)":  (CollinearGradients, dict(
            objective_fn=loss_fn, **MKG_COMMON,
        )),
    }


def run_experiment():
    """
    Перебирает (n, kappa), запускает старую и новую версии МКГ.

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
    Композитный рисунок 3x2 — сравнение старой и новой версий МКГ.
    """
    data_n = aggregate_by_n(results)
    data_k = aggregate_by_kappa(results)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    ax_a, ax_b = axes[0, 0], axes[0, 1]  # время
    ax_c, ax_d = axes[1, 0], axes[1, 1]  # итерации
    ax_e, ax_f = axes[2, 0], axes[2, 1]  # вычисления

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

    # (e) количество вычислений от n
    for name, d in data_n.items():
        ax_e.plot(d["n"], d["evals_mean"], marker="o", label=name)
    ax_e.set_xlabel("Размерность n")
    ax_e.set_ylabel(r"Вычислений $f + \nabla f$, шт.")

    # (f) количество вычислений от обусловленности
    for name, d in data_k.items():
        ax_f.plot(d["kappa"], d["evals_mean"], marker="o", label=name)
    ax_f.set_xlabel("Число обусловленности $\\kappa$")
    ax_f.set_ylabel(r"Вычислений $f + \nabla f$, шт.")

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
    out_path = RESULTS_DIR / "comparison_mkg_old_vs_new.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nКомпозитный рисунок сохранён: {out_path.resolve()}")


def print_speedup_table(results):
    """Печатает таблицу ускорения новая/старая по каждой (n, kappa)."""
    print("\n" + "=" * 90)
    print("Ускорение новой версии относительно старой (time_old / time_new):")
    print("=" * 90)
    header = f"{'n':>6} {'kappa':>6} {'old, s':>10} {'new, s':>10} {'speedup':>10} " \
             f"{'old_it':>8} {'new_it':>8} {'old_ev':>8} {'new_ev':>8}"
    print(header)
    print("-" * len(header))

    for n in N_VALUES:
        for kappa in KAPPA_VALUES:
            old = results["МКГ (старая)"][n][kappa]
            new = results["МКГ (новая)"][n][kappa]
            speedup = old["time"] / new["time"] if new["time"] > 1e-12 else float("inf")
            print(f"{n:>6} {kappa:>6} "
                  f"{old['time']:>10.4f} {new['time']:>10.4f} {speedup:>10.2f} "
                  f"{old['iters']:>8} {new['iters']:>8} "
                  f"{old['total_evals']:>8} {new['total_evals']:>8}")
        print("-" * len(header))

    # сводка
    all_old = sum(results["МКГ (старая)"][n][k]["time"]
                  for n in N_VALUES for k in KAPPA_VALUES)
    all_new = sum(results["МКГ (новая)"][n][k]["time"]
                  for n in N_VALUES for k in KAPPA_VALUES)
    print(f"\nСуммарное время: старая = {all_old:.2f} с, новая = {all_new:.2f} с, "
          f"ускорение = {all_old / all_new:.2f}×")

    # среднее по итерациям и вычислениям
    mean_it_old = np.mean([results["МКГ (старая)"][n][k]["iters"]
                           for n in N_VALUES for k in KAPPA_VALUES])
    mean_it_new = np.mean([results["МКГ (новая)"][n][k]["iters"]
                           for n in N_VALUES for k in KAPPA_VALUES])
    mean_ev_old = np.mean([results["МКГ (старая)"][n][k]["total_evals"]
                           for n in N_VALUES for k in KAPPA_VALUES])
    mean_ev_new = np.mean([results["МКГ (новая)"][n][k]["total_evals"]
                           for n in N_VALUES for k in KAPPA_VALUES])
    print(f"Средние итерации:  старая = {mean_it_old:.1f}, новая = {mean_it_new:.1f}")
    print(f"Средние вычисления: старая = {mean_ev_old:.1f}, новая = {mean_ev_new:.1f}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    results, total_times = run_experiment()
    print(f"\nВсе эксперименты заняли {time.perf_counter() - t0:.1f} с")

    print("\nСуммарное время решения всех задач:")
    for name, t in total_times.items():
        print(f"  {name:35s} {t:10.2f} с")

    print_speedup_table(results)
    plot_all(results)