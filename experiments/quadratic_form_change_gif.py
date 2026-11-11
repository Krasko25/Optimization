# gif_function_3d.py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio

# Настройка шрифтов для корректного отображения формул
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False

# ----------------------------- ПАРАМЕТРЫ -----------------------------
n_frames = 120
fps = 20
kappa_max = 20.0
grid_range = 2.0
n_grid = 80
output_gif = "surface_3d_hd_loop.gif"


def build_A(theta, kappa):
    c, s = np.cos(theta), np.sin(theta)
    U = np.array([[c, -s], [s, c]])
    Lambda = np.diag([1.0, kappa])
    A = U @ Lambda @ U.T
    return 0.5 * (A + A.T)


x = np.linspace(-grid_range, grid_range, n_grid)
y = np.linspace(-grid_range, grid_range, n_grid)
X, Y = np.meshgrid(x, y)


def quadratic_value_2d(X, Y, A, b):
    quadratic = 0.5 * (A[0, 0] * X ** 2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y ** 2)
    linear = b[0] * X + b[1] * Y
    return quadratic - linear


b = np.array([0.0, 0.0])

frames = []
print("Генерация кадров...")

for i in range(n_frames):
    # Угол делает полный оборот
    theta = 2.0 * np.pi * i / (n_frames - 1)

    # Kappa меняется по косинусу
    t = i / (n_frames - 1)
    kappa = 1.0 + (kappa_max - 1.0) * (0.5 - 0.5 * np.cos(2 * np.pi * t))

    A = build_A(theta, kappa)
    Z = quadratic_value_2d(X, Y, A, b)

    fig = plt.figure(figsize=(12, 9), dpi=300, facecolor='white')
    ax = fig.add_subplot(111, projection="3d")

    ax.computed_zorder = False

    ax.plot_surface(
        X, Y, Z,
        cmap="Oranges",
        linewidth=0,
        antialiased=True,
        rstride=2, cstride=2,
        alpha=1.0,
        zorder=1,
    )

    ax.scatter([0], [0], [0], color="red", s=100, zorder=10)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    ax.set_xlabel(r'$u_1$', fontsize=24, labelpad=15, rotation=0)
    ax.set_ylabel(r'$u_2$', fontsize=24, labelpad=15, rotation=0)
    ax.set_zlabel(r'$f(u_1, u_2)$', fontsize=24, labelpad=40, rotation=0)

    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)

    ax.set_box_aspect((1, 1, 0.8))
    ax.view_init(elev=30, azim=-75)

    ax.set_title(f"θ = {np.degrees(theta):.0f}°,  κ = {kappa:.1f}", fontsize=18)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frames.append(img)
    plt.close(fig)

    if (i + 1) % 10 == 0:
        print(f"  кадр {i + 1}/{n_frames}")

print(f"Сохранение {output_gif}...")
# loop=0 означает бесконечное зацикливание
imageio.mimsave(output_gif, frames, fps=fps, loop=0)
print("Готово.")