from src.optimization.generators import create_and_save_problem
from pathlib import Path

if __name__ == "__main__":
    # 2-10 с шагом 1, 20-100 с шагом 10, 150-1000 с шагом 50
    dims = [2, 3, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 1000, 2000, 5000, 10000]
    kappas = range(1, 21)
    R = 5.0
    NUM_REPEATS = 1  # Сколько случайных задач генерировать для каждого kappa

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