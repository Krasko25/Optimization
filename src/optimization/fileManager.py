import numpy as np
from pathlib import Path
from optimization.generators import generate_problem_data


def create_and_save_problem(n: int, kappa: float, R: float = 5.0,
                            seed_A: int | None = None,
                            seed_b: int | None = None,
                            seed_u0: int | None = None,
                            save_dir: Path | str = "data/quadratics") -> None:
    """Генерирует задачу и сохраняет её в указанную папку."""
    # Получаем данные
    data = generate_problem_data(n, kappa, seed_A, seed_b, seed_u0, R)

    # Приводим путь к объекту Path
    save_dir = Path(save_dir)
    # Создаем папку (и все родительские папки), если её нет
    save_dir.mkdir(parents=True, exist_ok=True)

    # Формируем имя файла и сохраняем
    filename = save_dir / f"n{n}_k{kappa}_s{data['seed_A'][0]}.npz"
    np.savez(filename, **data)

    print(f"Сохранено: {filename}")


def load_problem_file(filepath: str | Path) -> tuple[np.ndarray, ...]:
    """Загружает матрицу A, векторы b, u0 и u_star из файла .npz."""
    data = np.load(filepath)
    return data['A'], data['b'], data['u0'], data['u_star']