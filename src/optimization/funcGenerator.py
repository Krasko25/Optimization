import numpy as np

def generate_quadratic(kappa: float, n: int = 2, seed: int | None = None) -> np.ndarray:
    """
    Генерирует симметричную положительно определённую матрицу A
    с заданным числом обусловленности kappa.
    """
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

def quadratic_value(X: np.ndarray, Y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Вычисляет значения функции f(x1, x2) = 0.5 * x^T A x
    для 2D-матрицы A.
    """
    # Для 2D случая расписываем явно для скорости
    return 0.5 * (A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2)