import numpy as np

def quadratic_value_2d(X, Y, A):
    """Для 2D-графиков. Работает быстро, но только при n=2"""
    return 0.5 * (A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2)


def quadratic_value_nd(x, A):
    """Для N-мерной оптимизации. Универсально, но для 2-мерного случая лучше quadratic_value_2."""
    return 0.5 * x @ A @ x