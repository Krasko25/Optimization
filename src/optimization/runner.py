import torch
import time
import numpy as np


def run_optimization(optimizer_class, A_np, b_np, u0_np, u_star_np, num_iters=5000, tol=0.05, **opt_kwargs):
    """
    Универсальный цикл оптимизации для любых методов в стиле PyTorch
    """
    # Конвертируем в тензоры PyTorch
    A = torch.tensor(A_np, dtype=torch.float64)
    b = torch.tensor(b_np, dtype=torch.float64)
    u_star = torch.tensor(u_star_np, dtype=torch.float64)

    # 2. Создаем параметр и оптимизатор
    u = torch.tensor(u0_np, dtype=torch.float64, requires_grad=True)
    optimizer = optimizer_class([u], **opt_kwargs)

    dist0 = torch.norm(u - u_star).item()
    hist = {'f': [], 'dist': [], 'time': 0.0, 'iters': 0}

    start_time = time.perf_counter()

    for t in range(num_iters):
        # 3. Стандартный цикл PyTorch
        optimizer.zero_grad()

        # Вычисляем функцию потерь (и градиент автоматически)
        loss = 0.5 * u @ A @ u - b @ u
        loss.backward()

        # Шаг метода
        optimizer.step()

        # 4. Считаем метрики (без градиента)
        with torch.no_grad():
            f_val = loss.item()
            dist = torch.norm(u - u_star).item()
            hist['f'].append(f_val)
            hist['dist'].append(dist)

            if dist / dist0 < tol:
                break

    hist['time'] = time.perf_counter() - start_time
    hist['iters'] = t + 1

    return u.detach().numpy(), hist