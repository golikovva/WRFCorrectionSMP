# Seasonal ANO: модель и learning curves

ANO прибавляет к WRF среднюю обучающую ошибку `ERA5_on_WRF - WRF` в каждой ячейке и канале. В сезонном режиме поправка отдельная для каждой группы месяцев. Все часы группы имеют одинаковый вес, в том числе если количество обучающих часов в разных месяцах отличается.

## Настройка модели

```python
from lib.models.ano_corr_accumulator import ANOCorrector

model = ANOCorrector(
    time_mean_period="season",  # Также принимается "seasonal".
    season_months=[[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
)
```

Это стандартные группы DJF, MAM, JJA, SON. Можно задать другое количество групп: все месяцы 1–12 должны встречаться ровно один раз, пустые группы запрещены. Порядок групп определяет индексы сезонных полей. Настройка меняет усреднение ANO, но не календарные сезоны функции выборки weeks per season.

Модель учитывает месяц **каждого часа** последовательности, включая переходы между сезонами и годами. `day`, `month`, `all`, прежние алиасы и числовые индексы периодов сохраняются. Необученный период возвращает нулевую поправку; эксперимент learning curve дополнительно запрещает незаполненные сезонные группы.

```python
model.save_correction_fields("correction_fields.npz")
restored = ANOCorrector.from_correction_fields("correction_fields.npz")
```

NPZ содержит суммы, счётчики и конфигурацию модели, включая группы месяцев и шаг времени. Метаданные хранятся как JSON, без pickle. Явно указанные параметры, противоречащие сохранённым, вызывают ошибку. Загрузка на модель без явных ограничений восстанавливает параметры из архива. Старые NPY содержат только средние поля: при их загрузке период и нестандартные группы нужно задавать явно, а счётчики восстанавливаются как единицы.

Существующие команды обучения ANO принимают сезонные параметры:

```text
--mean-period season --season-months 12,1,2 3,4,5 6,7,8 9,10,11
```

Аргументы доступны как в `main.py`, так и в `sample_size_cv.py`. Для seasonal они сохраняют NPZ, для прежних режимов — NPY. `sample_size_cv.py` по-прежнему выполняет blocked temporal CV; для вложенных weeks per season используйте новый `train_size.py`.

## Протокол статьи

Отдельная конфигурация: `/home/configs/ano_train_size.yaml`. Она не наследует текущую конфигурацию нейросетевого обучения.

| Параметр | Значение |
| --- | --- |
| Регион | Nestp |
| Диапазон | [2019-01-01, 2020-12-31), часовой шаг |
| Train | 2019, 1–9 недельных блоков на каждый календарный сезон |
| Validation | 1 фиксированный блок по 14 дней на сезон, исключён из train |
| Test | 2020-01-01 00:00 — 2020-12-30 23:00 включительно |
| Train / validation seed | 42 / 42 |
| Длина последовательности | 4 часа, без перекрытий |
| Размеры | Все целые 1, 2, 3, 4, 5, 6, 7, 8, 9 wps |
| Метрики графиков | LAT_WEIGHTED_MAE для u10, v10, t2m на ERA-сетке |

Правая граница исключается, как в существующем эксперименте статьи: **31 декабря 2020 не входит в тест**. Для каждого размера train содержит `672 × wps` часов, validation — 1 344, test — 8 760. Все train-выборки вложенные; validation и test одинаковы. Один seed, без доверительных интервалов.

Поправка каждого размера вычисляется заново только по его train. Validation не используется для подбора модели. WRF и ANO оцениваются одним общим `lib.pipeline.test.test`, как нейросетевые модели; ANO использует физические единицы без дополнительного масштабирования.

## Запуск в Docker

Все команды выполняются из корня репозитория через работающий Compose service `dev`, внутри контейнера рабочая папка — `/home`. Локальные проверки используют CPU и один поток. Не запускайте CUDA-вариант одновременно с Triton-бенчмарками на том же GPU.

Проверка всех девяти разбиений **без метеоданных**:

```powershell
docker compose exec -T -w /home dev env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg python -m experiments.ano_baseline.train_size --config /home/configs/ano_train_size.yaml --output-dir /home/logs/ano_nestp_season_cpu --dry-run
```

Перед реальным серверным расчётом настройте пути `data` в отдельной конфигурации. WRF и ERA должны быть доступны **внутри dev**. По умолчанию нужны:

- `/home/wrf_nestp/`: файлы WRF с `uvmet10` и `T2`;
- `/home/era_data/uv10m/`: месячные ERA5 u10/v10;
- `/home/era_data/t2/`: месячные ERA5 t2m.

`data.era_wind_subdir` задаётся явно как `uv10m` в новом эксперименте. Существующие ANO-конфигурации без этого поля продолжают использовать `w10`. Станции и scatter не нужны.

Расчёт на CPU после dry-run с теми же параметрами:

```powershell
docker compose exec -T -w /home dev env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg python -m experiments.ano_baseline.train_size --config /home/configs/ano_train_size.yaml --output-dir /home/logs/ano_nestp_season_cpu
```

Расчёт на отдельном серверном GPU, если он свободен:

```powershell
docker compose exec -T -w /home dev env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg python -m experiments.ano_baseline.train_size --config /home/configs/ano_train_size.yaml --device cuda --output-dir /home/logs/ano_nestp_season_cuda
```

По умолчанию используется CPU, для него загрузка данных идёт без дополнительных workers. `--device cuda` — явное включение GPU; отсутствие CUDA вызывает ошибку. Можно предварительно выполнить `--device cuda --dry-run`: этот режим не обращается к GPU. При отсутствии `--output-dir` создаётся новая папка с UTC-временем в `data.logs_folder/ano_train_size/`.

`--weeks-per-season 1 2 3 4 5 7 9` позволяет выбрать подмножество размеров, например для сопоставления с прежним графиком. Стандартный запуск включает также 6 и 8 wps.

Непустой каталог запуска не перезаписывается. Исключение — повторный запуск из dry-run с **идентичной** конфигурацией и списком размеров. При изменении путей данных, устройства или других параметров задайте новую выходную папку. Автоматического продолжения частично выполненного расчёта нет; неуспешный запуск сохраняется со статусом `failed` и причиной.

## Результаты и сравнение с нейросетями

В каждой папке `1wps`…`9wps`:

- `era_metrics_table.csv`: стандартные метрики на WRF-сетке;
- `mean_era_metrics_table.csv`: стандартные метрики на ERA-сетке, включая LAT_WEIGHTED_MAE;
- `experiment_metrics.pickle`: тот же формат результатов, что у нейросетевого тестирования;
- `correction_fields.npz`, `season_counts.json`: модель и фактические сезонные счётчики;
- `split_dates.npz`, `split_manifest.json`: даты train/validation/test и начала последовательностей;
- `coverage.json`: фактически обработанные часы и последовательности train/test.

В таблицах имена моделей — **ANO** и **orig**; на графиках они подписаны **ANO seasonal** и **WRF**. Имена колонок, набор стандартных метрик, интерполяция и агрегирование совпадают с общим тестовым pipeline. Температурный канал в стандартных CSV называется **t2**, на итоговых графиках — **t2m**. Округление таблиц-картинок не применяется к значениям CSV.

В корне запуска сохраняются `resolved_config.yaml`, `run_manifest.json`, общий `split_manifest.json`, три пары файлов `lat_weighted_mae_{u10,v10,t2m}.png/.pdf` и `summary.csv`. Summary содержит 54 строки при девяти размерах: `nwps, model, variable, metric, value, experiment, source_file`. Числа сохраняются с полной точностью.

Дополнительные станции, scatter, региональные отчёты, profiler, spectra и прогнозные поля отключены. Общий тестовый pipeline может сохранять свои стандартные картинки таблиц и карты ошибок; его реализация не изменена.

Повторное построение графиков из готовых таблиц, без WRF/ERA:

```powershell
docker compose exec -T -w /home dev env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg python -m experiments.ano_baseline.train_size --plot-only --output-dir /home/logs/ano_nestp_season_cuda
```

Для неполного набора размеров добавьте тот же `--weeks-per-season`. Plot-only обновляет только графики и summary, не меняя модель, исходные таблицы, конфигурацию или манифесты. Проверяется наличие ровно одной строки ANO и WRF для каждого размера, конечность ошибок и постоянство WRF между размерами.

Чтение отсутствующих или неполных WRF/ERA последовательностей, NaN/Inf, пропуски дат, повторные даты, незаполненные группы месяцев и несовпадение фактических сезонных счётчиков с разбиением вызывают ошибку. Неполные последовательности проверяются **до** стандартного collate, чтобы дополнение нулями не скрывало пропуски.

## Проверки

```powershell
docker compose exec -T -w /home dev env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -q tests/test_ano_corr_accumulator.py tests/test_ano_helpers.py tests/test_ano_train_size.py
```

Тесты используют небольшие искусственные данные: сезонные средние и границы календаря, совместимость и архивы, точные вложенные разбиения, строгую проверку покрытия, стандартные таблицы и графики. Они не заменяют расчёт реальных ошибок на серверных данных.
