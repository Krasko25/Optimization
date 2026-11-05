import torch

from optimization.generators import generate_problem_data
from optimization.losses import make_quadratic_loss
from optimization.runner import run_optimization


def main():
    # 1. Генерируем задачу
    data = generate_problem_data(n=10, kappa=100.0, seed_A=42, seed_b=43, seed_u0=44)

    # 2. Оборачиваем A, b в torch-loss
    loss_fn = make_quadratic_loss(data["A"], data["b"])

    # 3. Запускаем оптимизацию (SGD)
    u_final, hist = run_optimization(
        torch.optim.SGD,
        u0_np=data["u0"],
        loss_fn=loss_fn,
        u_star_np=data["u_star"],
        num_iters=5000,
        tol_dist=0.05,
        lr=0.01,
    )

    print(f"Итерации      = {hist['iters']}")
    print(f"Вычислений функции    = {hist['f_evals']}")
    print(f"Вычислений градиента = {hist['grad_evals']}")
    print(f"Время       = {hist['time']:.4f} s")
    print(f"Расстояние до u* = {hist['dist'][-1]:.6e}")


if __name__ == "__main__":
    main()