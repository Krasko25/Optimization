from src.optimization.generators import create_and_save_problem
from pathlib import Path

if __name__ == "__main__":
    # 2-10 с шагом 1, 20-100 с шагом 10, 150-1000 с шагом 50
    dims = list(range(2, 10)) + list(range(20, 101, 20))
    kappas = range(1, 20)
    R = 5.0
    NUM_REPEATS = 2  # Сколько случайных задач генерировать для каждого kappa

    # Находим корень проекта.
    # __file__ это путь к текущему скрипту.
    # .resolve().parent - поднимаемся на 2 уровня вверх
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Формируем полный путь до нужной папки
    SAVE_PATH = PROJECT_ROOT / "data" / "quadratics1"

    for n in dims:
        for kappa in kappas:
            for i in range(NUM_REPEATS):
                create_and_save_problem(
                    n=n,
                    kappa=kappa,
                    R=R,
                    save_dir=SAVE_PATH  # Передаем абсолютный путь
                )