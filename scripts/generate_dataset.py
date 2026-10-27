import numpy as np
from pathlib import Path
from src.optimization.generators import generate_quadratic


def create_and_save_problem(n: int, kappa: float, seed: int):
    """Генерирует матрицу, начальную точку и сохраняет в .npz"""
    # Генерируем матрицу
    A = generate_quadratic(kappa=kappa, n=n, seed=seed)
    # Генерируем начальную точку
    # Используем тот же seed, чтобы точка была воспроизводимой
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n) * 2.0  # умножаем на 2, чтобы отойти от нуля

    # Вектор b для f(x) = 0.5*x^T A x - b^T x
    b = rng.standard_normal(n)

    # Формируем путь и сохраняем
    save_dir = Path("data/quadratics")
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = save_dir / f"n{n}_k{kappa}.npz"

    # np.savez сохраняет словарь массивов
    np.savez(filename, A=A, b=b, x0=x0)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    # Генерируем сетку задач
    dims = [2, 10, 100]
    kappas = [1, 10, 100, 1000]

    for n in dims:
        for kappa in kappas:
            # В качестве seed можно использовать комбинацию n и kappa
            create_and_save_problem(n=n, kappa=kappa, seed=n + int(kappa))