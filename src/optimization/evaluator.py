import numpy as np


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