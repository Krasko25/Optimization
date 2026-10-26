import numpy as np
import matplotlib.pyplot as plt


def plot_surface(ax, X, Y, Z,
                 title: str, Length: float = 3.0):
    """Рисует одну 3D-функцию на переданной оси"""
    # 1 Опускаем пол на 5% размаха ниже минимума
    # Нужно, чтобы был запас внизу от нижней точки Z для тени
    z_min = Z.min()
    z_max = Z.max()
    # Высота всего графика
    z_range = z_max - z_min
    # новый пол
    z_off = z_min - 0.05 * z_range

    # 2 Создание поверхности фукнции
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.92,
                    rstride=3, cstride=3) # rstride и cstride значат отрисовка только каждой 3-й линии ради оптимизации
    # 3. Проекция на пол
    ax.contour(X, Y, Z, zdir='z', offset=z_off, levels=15,
               cmap='viridis', linewidths=0.9)
    # 4. заголовки осей
    ax.set_title(title, fontsize=11)
    ax.set(xlabel='x₁', ylabel='x₂', zlabel='f(x)',
           xlim=(-Length, Length), ylim=(-Length, Length), zlim=(z_off, Z.max()))
    # 5. положение камеры
    # elem - угол подъёма камеры над плоскостью
    # azim - угол поворота камеры вокруг вертикальной оси
    ax.view_init(elev=28, azim=-50)
