---
name: quality-guard
description: Агент-детектив качества данных. Находит пропуски, дубликаты, выбросы, дисбаланс классов — визуализирует каждую проблему, предлагает стратегию чистки, применяет после подтверждения, генерирует сравнительный отчёт.
license: MIT
metadata:
  category: data-engineering
  version: 1.0.0
---

# Quality Guard

Агент для выявления и устранения проблем качества данных.

## ВАЖНО

Все файлы в ТЕКУЩЕЙ рабочей директории.
Вход: `data/raw/combined.parquet` (результат data-collector).
Выход: `data/cleaned/cleaned.parquet`.

---

## Workflow

### Шаг 0: Проверка входных данных

```bash
.venv/bin/python -c "
import pandas as pd, os
path = 'data/raw/combined.parquet'
if not os.path.exists(path):
    print('ERROR: файл не найден. Сначала запусти /data-collector')
    exit(1)
df = pd.read_parquet(path)
print(f'Загружено: {len(df)} строк, {len(df.columns)} колонок')
print(f'Колонки: {list(df.columns)}')
print(df.dtypes)
"
```

---

### Шаг 1: Анализ проблем качества

```bash
mkdir -p data/detective
.venv/bin/python ~/.claude/skills/quality-guard/scripts/audit.py \
    --input data/raw/combined.parquet \
    --output data/detective
```

Скрипт создаёт:
- `data/detective/problems.json` — найденные проблемы с метриками
- `data/detective/missing_values.png`
- `data/detective/outliers.png`
- `data/detective/class_balance.png`
- `data/detective/duplicates.png`

---

### Шаг 2: HITL Checkpoint #2 — выбор стратегии чистки

Прочитать `data/detective/problems.json` и показать пользователю:

```
## Результаты аудита качества данных

| Проблема           | Количество | %     | Серьёзность |
|--------------------|-----------|-------|-------------|
| Пропуски (text)    | ...       | ...%  | ...         |
| Дубликаты          | ...       | ...%  | ...         |
| Выбросы (длина)    | ...       | ...%  | ...         |
| Дисбаланс классов  | ratio=... | —     | ...         |
| Пустые тексты      | ...       | ...%  | ...         |

Визуализации: data/detective/*.png

### Доступные стратегии чистки:

**aggressive** — максимальная чистота, меньше данных
- Пропуски: удалить строки
- Дубликаты: удалить все
- Выбросы: удалить по IQR
- Когда: данных много, важна чистота

**conservative** — сохранить как можно больше данных
- Пропуски: заполнить пустой строкой
- Дубликаты: оставить
- Выбросы: оставить
- Когда: данных мало, каждый пример важен

**balanced** — компромисс (рекомендуется)
- Пропуски: заполнить, пустые — удалить
- Дубликаты: удалить
- Выбросы: удалить экстремальные (z-score > 3)
- Когда: стандартный сценарий

Какую стратегию применить? [aggressive / conservative / balanced]:
```

**ЖДАТЬ ОТВЕТА ПОЛЬЗОВАТЕЛЯ.**

После выбора — объяснить почему эта стратегия подходит для конкретной ML-задачи (1-2 предложения).

---

### Шаг 3: Применение стратегии

```bash
mkdir -p data/cleaned
.venv/bin/python ~/.claude/skills/quality-guard/scripts/cleaner.py \
    --input data/raw/combined.parquet \
    --output data/cleaned/cleaned.parquet \
    --strategy <STRATEGY>
```

---

### Шаг 4: Сравнительный отчёт

```bash
.venv/bin/python ~/.claude/skills/quality-guard/scripts/reporter.py \
    --before data/raw/combined.parquet \
    --after data/cleaned/cleaned.parquet \
    --problems data/detective/problems.json \
    --strategy <STRATEGY> \
    --output data/detective/comparison.md
```

Показать таблицу сравнения до/после:

```
## Сравнение до/после — стратегия: <STRATEGY>

| Метрика            | До      | После   | Изменение |
|--------------------|---------|---------|-----------|
| Всего строк        | 10,000  | 9,200   | -8%       |
| Пропуски           | 150     | 0       | -100%     |
| Дубликаты          | 500     | 0       | -100%     |
| Выбросы            | 200     | 12      | -94%      |
| Дисбаланс (ratio)  | 4.2x    | 4.2x    | —         |
```

---

## Итог шага

- `data/cleaned/cleaned.parquet` — очищенный датасет
- `data/detective/problems.json` — найденные проблемы
- `data/detective/comparison.md` — сравнение до/после
- `data/detective/*.png` — визуализации

---

## Правила

1. Всегда показывать все 4 типа проблем (пропуски, дубли, выбросы, дисбаланс)
2. ОБЯЗАТЕЛЬНО ждать подтверждения стратегии у пользователя
3. Обосновать выбор стратегии применительно к конкретной ML-задаче
4. Всегда показывать сравнительный отчёт до/после
5. Не пропускать шаг визуализации
