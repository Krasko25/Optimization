import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import time
import re


# загрузка файлов тестов
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "quadratics1"


def load_problem_file(filepath: str):
    data = np.load(filepath)
    return data['A'], data['b'], data['u0'], data['u_star']


# создание задачи
def compute_metrics(u, A, b, u_star):
    f = 0.5 * u @ A @ u - b @ u
    grad = A @ u - b
    return f, np.linalg.norm(grad), np.linalg.norm(u - u_star)


def nesterov(A, b, u0, u_star, num_iters=5000, tol=0.05):
    eigvals = np.linalg.eigvalsh(A)
    L, mu = eigvals[-1], eigvals[0]
    eta = 1.0 / L
    beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))

    u = u0.copy()
    y = u0.copy()
    hist = {'f': [], 'grad': [], 'dist': []}

    dist0 = np.linalg.norm(u0 - u_star)

    for _ in range(num_iters):
        grad_y = A @ y - b
        u_new = y - eta * grad_y
        y = u_new + beta * (u_new - u)
        u = u_new

        f, gn, d = compute_metrics(u, A, b, u_star)
        hist['f'].append(f)
        hist['grad'].append(gn)
        hist['dist'].append(d)

        if d / dist0 < tol:
            break
    return u, hist


def adam(A, b, u0, u_star, num_iters=5000, tol=0.05, lr=0.01, beta1=0.9, beta2=0.999, eps_adam=1e-8):
    u = u0.copy()
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    hist = {'f': [], 'grad': [], 'dist': []}

    dist0 = np.linalg.norm(u0 - u_star)

    for t in range(1, num_iters + 1):
        g = A @ u - b
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        u = u - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        f, gn, d = compute_metrics(u, A, b, u_star)
        hist['f'].append(f)
        hist['grad'].append(gn)
        hist['dist'].append(d)

        if d / dist0 < tol:
            break
    return u, hist


# Тестирование
all_files = glob(str(DATA_DIR / "*.npz"))
pairs = set()
for f in all_files:
    match = re.search(r"n(\d+)_k(\d+)_", Path(f).name)
    if match:
        pairs.add((int(match.group(1)), int(match.group(2))))

print(f"Найдено уникальных задач (n, kappa): {len(pairs)}")
print(f"{'n':<5} {'kappa':<8} {'Итер. Nest':<12} {'Итер. Adam':<12} {'Время Nest':<12} {'Время Adam':<12}")
print("-" * 80)

results = []

for n, kappa in sorted(pairs):
    pattern = str(DATA_DIR / f"n{n}_k{kappa}_s*.npz")
    file_to_load = sorted(glob(pattern))[0]

    A, b, u0, u_star = load_problem_file(file_to_load)
    f_star = -0.5 * b @ u_star

    # Замеряем Nesterov
    t0 = time.perf_counter()
    _, hist_nest = nesterov(A, b, u0, u_star, num_iters=5000, tol=0.05)
    time_nest = time.perf_counter() - t0

    # Замеряем Adam
    t0 = time.perf_counter()
    _, hist_adam = adam(A, b, u0, u_star, num_iters=5000, tol=0.05, lr=0.01)
    time_adam = time.perf_counter() - t0

    iter_nest = len(hist_nest['f'])
    iter_adam = len(hist_adam['f'])

    results.append({
        'n': n, 'kappa': kappa,
        'time_nest': time_nest, 'time_adam': time_adam,
        'iter_nest': iter_nest, 'iter_adam': iter_adam
    })

    print(f"{n:<5} {kappa:<8} {iter_nest:<12} {iter_adam:<12} {time_nest:<12.4f} {time_adam:<12.4f}")

# 4. Графики
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- График 1: Среднее время от размерности n ---
ns = sorted(list(set(r['n'] for r in results)))
mean_t_nest_n, std_t_nest_n = [], []
mean_t_adam_n, std_t_adam_n = [], []

for n in ns:
    # Собираем все результаты для конкретного n
    subset = [r for r in results if r['n'] == n]
    t_nest = [r['time_nest'] for r in subset]
    t_adam = [r['time_adam'] for r in subset]

    mean_t_nest_n.append(np.mean(t_nest))
    std_t_nest_n.append(np.std(t_nest))
    mean_t_adam_n.append(np.mean(t_adam))
    std_t_adam_n.append(np.std(t_adam))

# Строим график со свечами
ax1.errorbar(ns, mean_t_nest_n, yerr=std_t_nest_n, fmt='o-', label='Nesterov', color='C0', capsize=3)
ax1.errorbar(ns, mean_t_adam_n, yerr=std_t_adam_n, fmt='s--', label='Adam', color='C1', capsize=3)

ax1.set_xlabel("Размерность n")
ax1.set_ylabel("Среднее время (секунды)")
ax1.set_title("Среднее время от размерности\n(усреднено для всех обусловленностей")
ax1.legend()
ax1.grid(True, alpha=0.3)


# Среднее время от обусловленности
kappas = sorted(list(set(r['kappa'] for r in results)))
mean_t_nest_k, std_t_nest_k = [], []
mean_t_adam_k, std_t_adam_k = [], []

for k in kappas:
    # Собираем все результаты для конкретного k
    subset = [r for r in results if r['kappa'] == k]
    t_nest = [r['time_nest'] for r in subset]
    t_adam = [r['time_adam'] for r in subset]

    mean_t_nest_k.append(np.mean(t_nest))
    std_t_nest_k.append(np.std(t_nest))
    mean_t_adam_k.append(np.mean(t_adam))
    std_t_adam_k.append(np.std(t_adam))

ax2.errorbar(kappas, mean_t_nest_k, yerr=std_t_nest_k, fmt='o-', label='Nesterov', color='C0', capsize=3)
ax2.errorbar(kappas, mean_t_adam_k, yerr=std_t_adam_k, fmt='s--', label='Adam', color='C1', capsize=3)

ax2.set_xlabel("Число обусловленности")
ax2.set_ylabel("Среднее время (секунды)")
ax2.set_title("Среднее время от обусловленности\n(усреднено по всем n)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()