import numpy as np
import matplotlib.pyplot as plt

# Все импорты в начале файла
from generators import generate_quadratic
from evaluator import quadratic_value_2d
from plotting import plot_surface


def main():
    kappas = [1, 2, 5, 10, 20, 50]
    L = 3.0
    n_points = 150

    # Создаем сетку
    x_range = np.linspace(-L, L, n_points)
    y_range = np.linspace(-L, L, n_points)
    X, Y = np.meshgrid(x_range, y_range)

    # Создаем фигуру
    fig = plt.figure(figsize=(18, 11))

    for i, kappa in enumerate(kappas, 1):
        # 1. Генерируем матрицу
        A = generate_quadratic(kappa=kappa, n=2, seed=42)

        # 2. Считаем значения Z
        Z = quadratic_value_2d(X, Y, A)

        # 3. Рисуем
        ax = fig.add_subplot(2, 3, i, projection='3d')
        plot_surface(ax, X, Y, Z, title=f"Kappa = {kappa}", Length=L)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()