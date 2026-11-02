import torch
# Optimizеr - базовый класс в PyTorch для оптимизаторов
# required нужен, чтобы пометить какой-то параметр как обязательный
from torch.optim.optimizer import Optimizer, required


class CollinearGradients(Optimizer):
    def __init__(self, params, lr=required):
        # params - веса
        if lr is not required and lr < 0.0:
            raise ValueError(f"Некорректный шаговый множитель: {lr}")
        defaults = dict(lr=lr)
        # super - обращение к родительскому классу
        # разделение весов на группы
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Выполняет один шаг оптимизации"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                #Если градиент не задан, то есть слой не использовался, то пропускаем
                if p.grad is None:
                    continue

                # сохраняем градиент для текущего тензора
                # Тут .data нужно, чтобы изменить число, не записывая изменения в историю
                grad = p.grad.data
                # вычисления метода
                # p.data.add_(grad, alpha=-lr)

        return loss