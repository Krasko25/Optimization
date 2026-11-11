import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False

output_gif = "optimization_zigzag.gif"
fps = 3

n_grid = 100
grid_range = 5.0

A = np.array([[2.5, 0.5],
              [0.5, 2.5]])

# Матрица с сильным оврагом
A = np.array([[5.0, 4.5],
              [4.5, 5.0]])
b = np.array([0.0, 0.0])

# Начальная точка
u_start = np.array([3.5, 3.5])
u_opt = np.linalg.solve(A, b)

# Параметры спуска
lr = 0.18
n_steps = 8

x = np.linspace(-grid_range, grid_range, n_grid)
y = np.linspace(-grid_range, grid_range, n_grid)
X, Y = np.meshgrid(x, y)


def f(u1, u2):
    return 0.5 * (A[0, 0] * u1 ** 2 + 2 * A[0, 1] * u1 * u2 + A[1, 1] * u2 ** 2) - (b[0] * u1 + b[1] * u2)


def grad_f(u):
    return A @ u - b


Z = f(X, Y)

path = [u_start]
u = u_start.copy()
for _ in range(n_steps):
    g = grad_f(u)
    u = u - lr * g
    path.append(u)
path = np.array(path)

# анимация
frames = []
print("Генерация кадров...")

for step in range(len(path)):
    fig = plt.figure(figsize=(12, 9), dpi=150, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    ax.computed_zorder = False

    surf = ax.plot_surface(X, Y, Z, cmap='Oranges', alpha=1.0,
                           linewidth=0, antialiased=True,
                           zorder=1)
    if step > 0:
        ax.plot(path[:step + 1, 0], path[:step + 1, 1],
                f(path[:step + 1, 0], path[:step + 1, 1]),
                color='blue', linewidth=3.0, zorder=10)
        ax.scatter(path[:step + 1, 0], path[:step + 1, 1],
                   f(path[:step + 1, 0], path[:step + 1, 1]),
                   color='white', edgecolors='blue', s=70, linewidth=2.5,
                   zorder=11)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    ax.set_xlabel(r'$u_1$', fontsize=24, labelpad=15, rotation=0)
    ax.set_ylabel(r'$u_2$', fontsize=24, labelpad=15, rotation=0)
    ax.set_zlabel(r'$f(u_1, u_2)$', fontsize=24, labelpad=40, rotation=0)

    # Убираем панели и сетку
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_zlim(0, np.max(Z) * 1.1)

    # Пропорции и угол обзора
    ax.set_box_aspect((1, 1, 0.8))
    ax.view_init(elev=30, azim=-75)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frames.append(img)
    plt.close(fig)

    print(f"  кадр {step + 1}/{len(path)}")

print(f"Сохранение {output_gif}...")

imageio.mimsave(output_gif, frames, fps=fps, loop=0)

print("Готово.")