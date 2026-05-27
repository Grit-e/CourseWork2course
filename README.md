Проект содержит код генерации и анализа данных, обучение INN-модели, проверку предсказанных параметров и построение дисперсионных кривых, а также несколько экспериментов с PINN.

## Что делает проект

Основной сценарий работы:

1. Генерируются или загружаются данные с физическими параметрами пластины и соответствующими запрещёнными зонами.
2. Обучается обратимая нейронная сеть `INN`, которая связывает:
   - физические параметры системы;
   - границы запрещённой зоны частот.
3. По заданному диапазону частот, например `150–200 Hz`, модель предсказывает параметры:
   - `E` — модуль Юнга;
   - `nu` — коэффициент Пуассона;
   - `rho` — плотность;
   - `h` — толщина пластины;
   - `a` — размер ячейки;
   - `mR` — масса резонатора;
   - `fR_target` — целевая резонансная частота.
4. Для полученных параметров строится дисперсионная диаграмма и проверяется наличие band gap.

## Структура проекта

```text
CourseWork2course/
├── checkpoints/              # сохранённые веса моделей
├── data/
│   ├── datasets/             # Dataset и scaler для PyTorch
│   └── generated_data/       # train/val/test CSV-данные
├── graphics/                 # графики и визуальные материалы
├── notebooks/                # Jupyter-notebooks с экспериментами
├── papers/                   # статьи и материалы по теме
├── src/
│   ├── models/
│   │   └── INN.py            # архитектура invertible neural network
│   ├── train/
│   │   └── training.py       # обучение модели
│   └── test/
│       └── check_band_gaps.py # проверка band gap и визуализация
└── README.md
```

## Ноутбуки

В папке `notebooks/` находятся основные этапы исследования:

```text
01_formula_checks.ipynb 
02_data_generation.ipynb
03_formula_checks2.ipynb
04_data_analyze.ipynb
05_PINN_check.ipynb
05_PINN_datasphere.ipynb
```

Рекомендуемый порядок просмотра:

1. `01_formula_checks.ipynb` и `03_formula_checks2.ipynb` — проверка физических формул
2. `02_data_generation.ipynb` — генерация данных
3. `04_data_analyze.ipynb` — анализ полученных данных
4. `05_PINN_check.ipynb` и `05_PINN_datasphere.ipynb` — эксперименты с PINN


## Обучение модели

Для запуска обучения выполните:

```bash
python src/train/training.py
```

Параметры обучения задаются в классе `TrainConfig`:

```python
@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 128
    hidden_layers: int = 4
    num_blocks: int = 6
    clamp_scale: float = 2.0
    normalize_X: bool = True
    normalize_y: bool = True
```

Во время обучения сохраняется лучший checkpoint по validation loss.

По умолчанию файл сохраняется в:

```text
checkpoints/inn_dim8_best2.pth
```

## Проверка модели

Для проверки предсказания параметров по заданной запрещённой зоне можно запустить:

```bash
python src/test/check_band_gaps.py
```

В конце файла есть пример:

```python
check_band_gaps(
    f_low=150.0,
    f_high=200.0,
    title="Querry band gap (150hz, 200hz)"
)
```

Функция:

1. загружает checkpoint модели;
2. строит обратный проход `model.inverse(...)`;
3. получает физические параметры;
4. пересчитывает band gap физическим решателем;
5. строит дисперсионную диаграмму.

Если checkpoint после обучения называется `inn_dim8_best2.pth`, а в проверочном скрипте ожидается `inn_dim8_best.pth`, передайте путь явно:

```python
check_band_gaps(
    f_low=150.0,
    f_high=200.0,
    checkpoint_path="checkpoints/inn_dim8_best2.pth",
    title="Query band gap (150hz, 200hz)"
)
```

## Архитектура модели

Модель реализована в `src/models/INN.py`.

Основные компоненты:

- `STNet` — вспомогательная полносвязная сеть для scale/shift-преобразований;
- `InvertibleBlock` — обратимый affine coupling block;
- `PermutationLayer` — случайная перестановка координат между блоками;
- `INN` — последовательность обратимых блоков и перестановок.

Модель поддерживает два направления:

```python
y = model(x)          # прямое отображение: параметры -> band gap
x = model.inverse(y)  # обратное отображение: band gap -> параметры
```

## Физическая проверка

В `src/test/check_band_gaps.py` реализованы функции:

- `find_band_gaps(...)` — поиск запрещённых зон по массиву частот;
- `get_band_gap(...)` — расчёт band gap для заданных физических параметров;
- `get_dispersion_plot(...)` — построение дисперсионной диаграммы;
- `predict_parameters_from_band_gap(...)` — предсказание параметров через inverse INN;
- `check_band_gaps(...)` — полный pipeline проверки.

## Пример использования из Python

```python
from src.test.check_band_gaps import check_band_gaps

params = check_band_gaps(
    f_low=150.0,
    f_high=200.0,
    checkpoint_path="checkpoints/inn_dim8_best2.pth",
    title="Band gap 150-200 Hz"
)

print(params)
```
## Возможные проблемы

### `ModuleNotFoundError`

Запускайте скрипты из корня проекта:

```bash
cd CourseWork2course
python src/train/training.py
```

## Автор

Grigorij Evgenev  
Курсовая работа, 2 курс.

