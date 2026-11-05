import torch
import time
import numpy as np


def run_optimization(optimizer_class, u0_np, loss_fn, u_star_np=None,
                     num_iters=5000, tol_dist=0.05, tol_grad=1e-6, **opt_kwargs):
    """
    Универсальный цикл оптимизации для любых методов в стиле PyTorch
    """
    u = torch.tensor(u0_np, dtype=torch.float64, requires_grad=True)
    # Глобальный минимум, в нём градиент вычислять не нужно будет
    if u_star_np is not None:
        u_star = torch.tensor(u_star_np, dtype=torch.float64)
        # Начальное расстояние до оптимума (.item превращает тензор в число)
        dist0 = torch.norm(u - u_star).item()
    else:
        u_star = None
        dist0 = 1.0  # это заглушка, в этом случае dist0 никогда использоваться не будет

    # счётчики вызовов f и ∇f
    f_evals = 0
    grad_evals = 0

    def counted_loss_fn(u_):
        nonlocal f_evals # чтобы изменения внутри этой функции влияли на эту переменную вне функции
        f_evals += 1
        return loss_fn(u_)

    # экземпляр класса оптимизатора
    optimizer = optimizer_class([u], **opt_kwargs)
    hist = {'f': [], 'dist': [], 'grad_norm': [], 'time': 0.0, 'iters': 0,
            'f_evals': 0, 'grad_evals': 0}
    # этот таймер считает всё прошедшее время, а не только то, в которое процессор занимался этой программой
    # так как process_time суммировал бы время с каждого ядра, и perf_counter() является стандартом
    start_time = time.perf_counter()

    # каждая итерация - шаг оптимизации
    for i in range(num_iters):
        #обнуление градиента, чтобы не было суммы предыдущих
        optimizer.zero_grad()
        loss = counted_loss_fn(u)
        loss.backward()

        #счётчик градиента не заворачивается в функцию, так как .backward() - это
        # не пользовательская функция и для текущих методов этого будет достаточно
        grad_evals += 1

        if u.grad is not None:
            grad_norm = torch.norm(u.grad).item()
        else:
            grad_norm = 0.0

        optimizer.step()

        with torch.no_grad():
            f_val = loss.item()
            hist['f'].append(f_val)
            hist['grad_norm'].append(grad_norm)

            if u_star is not None:
                dist = torch.norm(u - u_star).item()
                hist['dist'].append(dist)
                if (dist / dist0) < tol_dist:
                    break
            elif grad_norm < tol_grad:
                break

    hist['time'] = time.perf_counter() - start_time
    hist['iters'] = i + 1
    hist['f_evals'] = f_evals
    hist['grad_evals'] = grad_evals

    # .detach() отвязывает тензор, .numpy преобразует в numpy формат
    return u.detach().numpy(), hist