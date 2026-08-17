from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    description: str


STAGES = [
    Stage(
        "load_graph",
        "1: Обработка графа",
        "Загрузка GeoJSON и проверка CRS",
    ),
    Stage(
        "tile_indices",
        "2: Вычисление тайлов",
        "Расчёт индексов тайлов для рёбер",
    ),
    Stage(
        "find_tiles",
        "3: Поиск тайлов",
        "Проверка локальных изображений",
    ),
    Stage(
        "prepare_tiles",
        "4: Подготовка тайлов",
        "Формирование рабочего набора",
    ),
    Stage(
        "predict_masks",
        "5: Предсказание масок",
        "Сегментация тайлов моделью",
    ),
    Stage(
        "calculate_widths",
        "6: Вычисление ширины",
        "Расчёт ширины по рёбрам",
    ),
    Stage(
        "update_graph",
        "7: Обновление графа",
        "Запись результата в GeoJSON",
    ),
]