# src/optimization/paths.py
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1) # после запуска программы хранить кэш пути
def get_project_root() -> Path:
    """
    Ищет корень проекта (папку с pyproject.toml), поднимаясь вверх.
    Использует кэширование (lru_cache), чтобы не сканировать диск при каждом вызове.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents): # сложение, чтобы искать в текущей тоже
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Не удалось найти корень проекта (pyproject.toml)")

def get_data_dir(folder_name: str = "quadratics1") -> Path:
    """Возвращает путь к папке с данными."""
    return get_project_root() / "data" / folder_name