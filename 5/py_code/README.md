# Симулятор навигации по магнитному маяку

Проект исследования и навигации по магнитному полю вращающегося диполя (магнитного маяка).

---

## Пример

Для запуска простейшего примера понадобиться uv. Запуск пример:
```sh
uv run scripts/simple_nav.py
```

---

### Установка дополнительных зависимостей

```sh
pip install -e '.[ml]'
```

### Запуск обучения

Базовая MLP-модель, групповой сплит по траекториям и фиксированный маяк:

```sh
python scripts/train_ml_corrector.py --model mlp
```

CatBoost с `MultiRMSE`:

```sh
python scripts/train_ml_corrector.py --model catboost --split-mode by_trajectory
```

Проверка обобщения на полностью новые конфигурации маяков (`beacon_id` не пересекаются между Train/Val/Test):

```sh
python scripts/train_ml_corrector.py --model mlp --split-mode by_beacon --beacons 10
```

Для быстрой проверки кода на маленьком наборе:

```sh
python scripts/train_ml_corrector.py --trajectories 6 --points 16 --epochs 3 --patience 2 --quiet --run-name smoke
```

Зерно генератора с лучайных числе по умолчанию фиксирован на `42`.


### Артефакты каждого запуска

Результаты создаются в `runs/<run_id>/`.
Визуализацию можно повторно построить по сохранённым CSV без повторного обучения:

```sh
python scripts/visualize_ml_training.py runs/<run_id>
```
