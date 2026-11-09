import math
import torch
from torch.optim.optimizer import Optimizer


class CollinearGradients(Optimizer):
    """
    Параметры
    ---------
    objective_fn :
        loss_fn(u). Если None — нужно передать closure
    lr : float
        Множитель шага, в этом методе он не нужен, так что это просто заглушка
    c1 : float
        Точность коллинеарности (по умолчанию 1e-8)
    c2 : float
        Параметр ограничения числа подитераций (по умолчанию 4.0)
    delta0 : float
        Начальный радиус окрестности (по умолчанию 1e-5)
    delta_m_factor : float
        Минимальный радиус = delta_m_factor * delta0 (по умолчанию 1e-15)
    h0 : float
        Шаг конечных разностей (по умолчанию 1e-8)
    l_max_cap : int
        Жёсткий потолок на число CG-подитераций (по умолчанию 1000)
    """

    def __init__(self, params, lr=1.0, objective_fn=None,
                 c1=1e-8, c2=4.0, delta0=1e-5, delta_m_factor=1e-15,
                 h0=1e-8, l_max_cap=1000):
        if lr < 0.0:
            raise ValueError(f"Шаговый множитель не может быть отрицательным {lr}")
        defaults = dict(lr=lr)

        # возвращает прокси на родительский класс, чтобы были param_groups, state, zero_grad
        super().__init__(params, defaults)
        self.objective_fn = objective_fn
        self.c1 = c1
        self.c2 = c2
        self.delta0 = delta0
        self.delta_m_factor = delta_m_factor
        self.h0 = h0
        self.l_max_cap = l_max_cap

        # Счётчики в дополнение к счётчикам runner'а
        self.f_evals = 0
        self.grad_evals = 0

    # сплющивание всех параметров в один плоский список
    # нужен так как pytorch работает с группами параметров, а МКГ с одним вектором u
    def _params_list(self):
        result = []
        # разные группы нужны, когда, например, разный lr к разным слоям нейросети
        for group in self.param_groups:  # одна группа (словарь)
            for param in group['params']:  # один тензор из этой группы
                result.append(param)
        return result

    # Превращение списка тензоров в вектор чисел
    def _get_flat(self):
        result = []
        # проходим по плоскому списку всех тензоров, который вернул _params_list
        for param in self._params_list():
            # .reshape(-1) — вытягиваем тензор любой формы (матрицу, куб) в одномерный вектор
            flat_param = param.data.reshape(-1)
            result.append(flat_param)
        # torch.cat cклеивает список одномерных векторов в один вектор u
        return torch.cat(result)

    # раскладывание плоского вектора u обратно по тензорам
    # flat куда попадает аргумент при вызове
    def _set_flat(self, flat):
        offset = 0  # текущая позиция (сдвиг) в плоском векторе
        for param in self._params_list():
            numEl = param.numel()  # сколько элементов в этом конкретном тензоре

            # flat[offset : offset + numel] — вырезаем из плоского вектора кусок нужной длины
            # .view_as(param) — превращаем этот одномерный кусок обратно в матрицу/тензор нужной формы
            # .copy_() — копируем значения прямо в param.data, не создавая новый тензор
            param.data.copy_(flat[offset: offset + numEl].view_as(param))

            offset += numEl  # сдвигаем позицию для следующего тензора

    # склеивание всех градиентов в один плоский вектор
    def _get_flat_grad(self):
        result = []
        for param in self._params_list():
            # если градиент для этого тензора еще не вычислен ,
            # то пропускаем его
            if param.grad is not None:
                # param.grad — это тензор градиента той же формы, что и param
                # .reshape(-1) — вытягиваем его в одномерный вектор
                flat_grad = param.grad.reshape(-1)
                result.append(flat_grad)

        # если список пустой, значит backward() еще не вызывали или ни один параметр не получил градиент
        # без градиента МКГ не может сделать шаг, поэтому выбрасываем ошибку
        if not result:
            raise RuntimeError("Градиенты не найдены — вызовите backward()")

        # torch.cat — склеивает все одномерные вектора градиентов в один
        return torch.cat(result)

    # вычисление градиента в произвольной точке u_flat
    def _eval_grad(self, u_flat):
        """Вычислить ∇J в u_flat. Считает f_evals и grad_evals."""

        # ставим параметры pytorch в нужную точку u_flat
        # внутри _set_flat происходит копирование значений
        self._set_flat(u_flat)

        # обнуляем старые градиенты
        for param in self._params_list():
            param.grad = None

        # получаем список тензоров
        params = self._params_list()

        # включаем режим autograd
        # нужен так как метод step() внутри которого вызывается МКГ обернут в @torch.no_grad(),
        # чтобы не строить граф для математических операций. Но для вычисления градиента
        # функции потерь граф нужен, поэтому мы принудительно его включаем
        with torch.enable_grad():
            # вычисляем значение функции потерь
            if len(params) == 1:
                loss = self.objective_fn(params[0])
            else:
                loss = self.objective_fn(*params)

            loss.backward()
        self.f_evals += 1
        self.grad_evals += 1

        # возвращаем градиент в виде плоского вектора
        # _get_flat_grad() — склеивает все градиенты в один вектор
        # .detach() — отвязывает тензор от графа autograd
        # .clone() — создает независимую копию
        return self._get_flat_grad().detach().clone()

    # вычисление невязки коллинеарности r(u) (формула 6 из статьи)
    def _residue(self, u_flat, grad_k, grad_k_norm):
        """Возвращает (∇J(u), r(u)) — градиент и невязку коллинеарности."""
        grad_u = self._eval_grad(u_flat)

        grad_u_norm = grad_u.norm()

        # защита от деления на ноль
        if grad_u_norm.item() < 1e-300:
            # torch.zeros_like создаёт такой же тензор, но заполненный нулями
            return grad_u, torch.zeros_like(grad_u)

        # вычисляем знак скалярного произведения двух градиентов
        # s = 1, если градиенты сонаправлены (угол < 90 градусов)
        # s = -1, если градиенты противоположно направлены (угол > 90 градусов)
        # что важно для поиска прямой, проходящей через оптимум
        s = 1.0 if (grad_u * grad_k).sum().item() >= 0 else -1.0

        # формула (6) из статьи:
        # r(u) = s * grad_u / ||grad_u|| - grad_k / ||grad_k||
        # если градиенты коллинеарны (сонаправлены или противоположны), то r = 0
        # если они перпендикулярны, то ||r|| = sqrt(2)
        r = s * grad_u / grad_u_norm - grad_k / grad_k_norm

        return grad_u, r

    # ------------------------------------------------------------------
    # Внешний шаг
    # ------------------------------------------------------------------
    # отключает построение вычисл графа autograd для операций внутри след. функции
    @torch.no_grad()    # Публичный метод, который вызывается извне (например, из runner.py).
    def step(self, closure=None):
        """Это "обертка" над _step_impl, которая решает вопросы совместимости и безопасности"""
        # защита от отсутствия функции
        if self.objective_fn is None and closure is None:
            raise RuntimeError(
                "МКГ требует objective_fn (в __init__) "
                "или closure (в step)"
            )
        original_fn = self.objective_fn

        # МКГ внутри _eval_grad ожидает функцию,
        # которая принимает аргументы (*args). closure() обычно аргументов не принимает
        # lambda *args: closure() позволяет игнорировать аргументы и просто вызвать closure
        if closure is not None and original_fn is None:
            self.objective_fn = lambda *args: closure()

        try:
            self._step_implementation()  # вся математика МКГ

        finally:
            # возвращаем objective_fn в исходное состояние.
            self.objective_fn = original_fn
        return None


    def _step_implementation(self):
        """
        Основная математика МКГ: одна внешняя итерация
        Состоит из двух этапов:
        1) Поиск точки u_k_star в окрестности u_k, где градиент коллинеарен исходному
        2) Вычисление направления d_k и шага b_k, ведущих к оптимуму, и обновление параметров
        """
        grad_k = self._get_flat_grad().clone()
        grad_k_norm = grad_k.norm()

        if grad_k_norm.item() < 1e-30:
            return

        # берем текущую точку u^k в виде плоского вектора
        u_k = self._get_flat().clone()
        n = u_k.numel()

        # Первая точка подитераций (формула 10 из статьи)

        # отступаем от u^k на расстояние delta_k в направлении роста функции
        # деление на sqrt(n) нужно, чтобы евклидова длина шага была ровно delta_k
        delta_k = self.delta0
        u_k_l = u_k + (delta_k / math.sqrt(n)) * torch.sign(grad_k)

        # Начальная невязка
        # _residue возвращает градиент и невязку
        grad_k_l, r_l = self._residue(u_k_l, grad_k, grad_k_norm)

        # храним квадрат нормы невязки
        r_l_norm_sq = (r_l * r_l).sum()

        # начальное направление CG — антиградиент невязки
        p_cg = -r_l.clone()

        # Настройка лимитов

        # ограничение числа подитераций по формуле: l_max = c2 * ln(n) * |ln(c1)|
        # max(n, 2) защищает от ln(1) = 0 (при n=1)
        # max(c1, 1e-15) защищает от ln(0) = -inf (при c1=0)
        # l_max_cap страховка от экстремальных настроек
        l_max = min(
            self.l_max_cap,
            max(2, round(self.c2 * math.log(max(n, 2))
                         * abs(math.log(max(self.c1, 1e-15))))),
        )

        # минимальный радиус окрестности: delta_m = delta_m_factor * delta_0
        # если МКГ уйдет от u_k дальше этого радиуса, поиск коллинеарности теряет смысл
        delta_m = self.delta_m_factor * self.delta0

        u_k_star = u_k_l.clone()
        grad_k_star = grad_k_l.clone()

        # поиск коллинеарной точки

        # цикл метода сопряженных градиентов на невязке r(u)
        for l in range(1, l_max + 1):

            # УСЛОВИЕ ОСТАНОВКИ 1: достигли нужной точности коллинеарности
            # ||r|| <= c1 * sqrt(2)
            if r_l_norm_sq.sqrt().item() <= self.c1 * math.sqrt(2):
                break

            # УСЛОВИЕ ОСТАНОВКИ 2: вышли за минимальный радиус окрестности
            if (u_k_l - u_k).norm().item() < delta_m:
                break

            # УСЛОВИЕ ОСТАНОВКИ 3: направление МКГ выродилось (почти нулевое)
            p_norm = p_cg.norm()
            if p_norm.item() < 1e-30:
                break

            # формула 9
            h = self.h0 / p_norm.item()
            u_pert = u_k_l + h * p_cg
            _, r_pert = self._residue(u_pert, grad_k, grad_k_norm)
            Hp = (r_pert - r_l) / h

            # скалярное произведение <p, Hp>
            pHp = (p_cg * Hp).sum()

            # защита от деления на ноль
            if pHp.abs().item() < 1e-300:
                continue

            # шаг МКШ (формула 8)
            # τ - длина шага
            tau = r_l_norm_sq / pHp  # оптимальный шаг МКГ
            u_k_l = u_k_l + tau * p_cg  # обновляем точку

            # новая невязка
            grad_k_l, r_l_new = self._residue(u_k_l, grad_k, grad_k_norm)
            r_l_new_norm_sq = (r_l_new * r_l_new).sum()

            # сохраняем эту точку как "лучшую"
            u_k_star = u_k_l.clone()
            grad_k_star = grad_k_l.clone()

            # Обновление направления МКГ (формула Флетчера-Ривса)
            # рестарт каждые n итераций (защита от потери сопряженности из-за шума конечных разностей)
            if l % n == 0:
                beta = 0.0  # сброс на обычный градиентный спуск
            else:
                beta = (r_l_new_norm_sq / max(r_l_norm_sq, 1e-300)).item()

            p_cg = -r_l_new + beta * p_cg  # новое направление
            r_l = r_l_new  # обновляем невязку
            r_l_norm_sq = r_l_new_norm_sq

        # Направление и шаг

        # направление d_k — это вектор из текущей точки в найденную коллинеарную точку
        d_k = u_k_star - u_k

        # если точка не сдвинулась (направление нулевое)
        if d_k.norm().item() < 1e-30:
            return

        # Вычисление оптимального шага b_k (формула 5)
        # b_k = 1 / (1 - <∇J(u^k), d^k> / <∇J(u^k_star), d^k>)
        # это минимум параболы f(u^k + b*d^k) вдоль направления d_k
        num = (grad_k * d_k).sum().item()
        den = (grad_k_star * d_k).sum().item()

        # защита от деления на ноль
        if abs(den) < 1e-300:
            return
        one_minus = 1.0 - num / den
        if abs(one_minus) < 1e-300:
            return

        b_k = 1.0 / one_minus

        # внешний множитель шага
        lr = self.param_groups[0]['lr']
        step_vec = lr * b_k * d_k

        # ПРОЦЕДУРА GETSTEP (принудительный спуск)
        # проверяем, что мы движемся в сторону УБЫВАНИЯ функции
        # скалярное произведение <-grad_k, step_vec> должно быть положительным (угол < 90 градусов)
        if (-grad_k * step_vec).sum().item() <= 0:
            # если мы движемся в гору — переворачиваем направление
            d_k = -d_k
            num = (grad_k * d_k).sum().item()
            den = (grad_k_star * d_k).sum().item()
            if abs(den) < 1e-300:
                return
            one_minus = 1.0 - num / den
            if abs(one_minus) < 1e-300:
                return
            b_k = 1.0 / one_minus
            step_vec = lr * b_k * d_k

        # обновление параметров
        # u^{k+1} = u^k + b^k * d^k
        u_new = u_k + step_vec

        # раскладываем плоский вектор обратно по pytorch-тензорам
        self._set_flat(u_new)