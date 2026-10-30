import numpy as np
from pathlib import Path

def generate_quadratic(kappa: float, n: int = 2, seed: int | None = None) -> np.ndarray:
    """
    Генерирует симметричную положительно определённую матрицу A
    с заданным числом обусловленности kappa.
    """
    if kappa < 1.0:
        raise ValueError("Число обусловленности kappa должно быть >= 1.0")

    rng = np.random.default_rng(seed)
    # Случайная ортогональная матрица (через QR)
    U, _ = np.linalg.qr(rng.standard_normal((n, n))) # среднее 0 дисперсия 1
    # Диагональ с собственными значениями [1; kappa]
    S = np.diag(np.linspace(1.0, kappa, n))
    # A = U * S * U^T
    A = U @ S @ U.T #U.T даёт транспонированную матрицу
    # Симметризация. Необходима из-за машинных ошибок, чтобы гарантировать идеальную симметрию
    # эта операция заменяет каждый элемент на среднее арифметическое между ним и его зеркалом
    return 0.5 * (A + A.T)

def generate_problem_data(n: int, kappa: float,
                          seed_A: int | None = None,
                          seed_b: int | None = None,
                          seed_u0: int | None = None,
                          R: float = 5.0) -> dict:
    """Генерирует данные задачи и возвращает их вместе с сидами"""
    # Если сиды не переданы, генерируем случайные
    rng_master = np.random.default_rng()
    if seed_A is None:
        seed_A = rng_master.integers(0, 2 ** 32 - 1)
    if seed_b is None:
        seed_b = rng_master.integers(0, 2 ** 32 - 1)
    if seed_u0 is None:
        seed_u0 = rng_master.integers(0, 2 ** 32 - 1)

    # Матрица A
    A = generate_quadratic(kappa=kappa, n=n, seed=seed_A)

    # Вектор b (норма 1.0)
    rng_b = np.random.default_rng(seed_b)
    b = rng_b.standard_normal(n)
    b = b / np.linalg.norm(b) * 1.0

    # Глобальный минимум u* = A^(-1)*b
    u_star = np.linalg.solve(A, b)

    # Начальная точка на расстоянии R от u_star
    rng_u0 = np.random.default_rng(seed_u0)
    direction = rng_u0.standard_normal(n)
    direction = direction / np.linalg.norm(direction)
    u0 = u_star + direction * R

    return {
        "A": A, "b": b, "u0": u0, "u_star": u_star,
        "seed_A": np.array([seed_A]),
        "seed_b": np.array([seed_b]),
        "seed_u0": np.array([seed_u0])
    }


def create_and_save_problem(n: int, kappa: float, R: float = 5.0,
                            seed_A: int | None = None,
                            seed_b: int | None = None,
                            seed_u0: int | None = None,
                            save_dir: Path | str = "data/quadratics") -> None:
    """Генерирует задачу и сохраняет её в указанную папку"""
    # Получаем данные
    data = generate_problem_data(n, kappa, seed_A, seed_b, seed_u0, R)

    # Приводим путь к объекту Path (на случай, если передали строку)
    save_dir = Path(save_dir)
    # Создаем папку (и все родительские папки), если её нет
    save_dir.mkdir(parents=True, exist_ok=True)

    # Формируем имя файла и сохраняем
    filename = save_dir / f"n{n}_k{kappa}_s{data['seed_A'][0]}.npz"
    np.savez(filename, **data)

    print(f"Сохранено: {filename}")

