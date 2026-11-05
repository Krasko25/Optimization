import numpy as np
import torch


def make_quadratic_loss(A_np: np.ndarray, b_np: np.ndarray):
    """
    Возвращает torch-совместимую loss_fn для f(u) = 0.5 * u^T A u - b^T u.

    A и b конвертируются в torch.float64 Один раз при создании замыкания
    (не на каждой итерации оптимизации). Полученная функция принимает
    torch-тензор u (float64) и возвращает n-мерный тензор — совместимо
    с loss.backward()
    """
    A = torch.tensor(A_np, dtype=torch.float64)
    b = torch.tensor(b_np, dtype=torch.float64)

    def loss_fn(u: torch.Tensor) -> torch.Tensor:
        return 0.5 * (u @ (A @ u)) - (b @ u)

    return loss_fn


def quadratic_value_2d(U1, U2, A, b):
    """
    Для 2D-графиков. Работает быстро (явная формула).
    U1, U2 — сетки координат.
    """
    # Квадратичная часть: 0.5 * u^T A u
    quadratic = 0.5 * (A[0, 0] * U1 ** 2 + 2 * A[0, 1] * U1 * U2 + A[1, 1] * U2 ** 2)
    # Линейная часть: - b^T u
    linear = b[0] * U1 + b[1] * U2

    return quadratic - linear


def quadratic_value_nd(u, A, b):
    """Для N-мерной оптимизации. Универсально, но для 2-мерного случая лучше quadratic_value_2d."""
    return 0.5 * u @ A @ u - b @ u