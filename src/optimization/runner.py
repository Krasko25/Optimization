import torch
import time
import numpy as np


def run_optimization(optimizer_class, u0_np, loss_fn, u_star_np=None,
                     num_iters=5000, tol_dist=0.05, tol_grad=1e-6, **opt_kwargs):
    """
    Универсальный цикл оптимизации для любых методов в стиле PyTorch
    """
    u = torch.tensor(u0_np, dtype=torch.float64, requires_grad=True)
    #Глобальный минимум, в нём градиент вычислять не нужно будет
    if u_star_np is not None:
        u_star = torch.tensor(u_star_np, dtype=torch.float64)
    else:
        u_star = None

    #Начальное расстояние до оптимума
    if u_star_np is not None:
        dist0 = torch.norm(u - u_star).item()
    else:
        dist0 = 1.0

    # счётчики вызовов f и ∇f
    f_evals = 0
    grad_evals = 0

    def counted_loss_fn(u_):
        nonlocal f_evals
        f_evals += 1
        return loss_fn(u_)

    optimizer = optimizer_class([u], **opt_kwargs)
    hist = {'f': [], 'dist': [], 'grad_norm': [], 'time': 0.0, 'iters': 0,
            'f_evals': 0, 'grad_evals': 0}
    start_time = time.perf_counter()

    for t in range(num_iters):
        optimizer.zero_grad()
        loss = counted_loss_fn(u)
        loss.backward()
        grad_evals += 1

        grad_norm = torch.norm(u.grad).item() if u.grad is not None else 0.0

        optimizer.step()

        with torch.no_grad():
            f_val = loss.item()
            hist['f'].append(f_val)
            hist['grad_norm'].append(grad_norm)

            if u_star is not None:
                dist = torch.norm(u - u_star).item()
                hist['dist'].append(dist)
                if dist / dist0 < tol_dist:
                    break
            elif grad_norm < tol_grad:
                break

    hist['time'] = time.perf_counter() - start_time
    hist['iters'] = t + 1
    hist['f_evals'] = f_evals
    hist['grad_evals'] = grad_evals
    return u.detach().numpy(), hist