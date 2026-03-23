---
name: smart-sampler
description: Active Learning агент. Итеративно обучает модель, выбирая наиболее информативные примеры для разметки (entropy/margin/random). Сравнивает стратегии, строит learning curves, показывает сколько примеров сэкономлено.
license: MIT
metadata:
  category: ml-training
  version: 1.0.0
---

# Smart Sampler — Active Learning

Агент для оптимизации процесса разметки через Active Learning.

## ВАЖНО

Все файлы в ТЕКУЩЕЙ рабочей директории.
Вход: `data/labeled/labeled_final.parquet` (результат auto-tagger).
Выход: `data/active/` — история, кривые обучения, отчёт.

---

## Workflow

### Шаг 0: Проверка входных данных

```bash
.venv/bin/python -c "
import pandas as pd, os
path = 'data/labeled/labeled_final.parquet'
if not os.path.exists(path):
    print('ERROR: файл не найден. Сначала запусти /auto-tagger')
    exit(1)
df = pd.read_parquet(path)
labeled = df[df['label'] != 'unlabeled']
print(f'Всего строк: {len(df)}')
print(f'Размечено: {len(labeled)}')
print(f'Классы: {labeled[\"label\"].value_counts().to_dict()}')
print(f'Минимум нужно: 100 размеченных примеров для AL')
"
```

---

### Шаг 1: HITL Checkpoint #4 — подтверждение настроек AL

Показать пользователю:

```
## Настройки Active Learning

- Датасет: data/labeled/labeled_final.parquet
- Размеченных: <N> примеров
- Seed (начальная выборка): 50 примеров
- Пул для AL: <N - 50 - test_size> примеров
- Тест: 20% датасета (фиксирован)
- Итераций: 5
- Batch per iteration: 20 примеров
- Модель: LogisticRegression
- Признаки: TF-IDF (text) / цветовые гистограммы (image) / MFCC (audio) — автоопределяется
- Стратегии: entropy, margin, random (сравниваем все три)

Подтверждаешь? [да / нет]:
```

**ЖДАТЬ ОТВЕТА ПОЛЬЗОВАТЕЛЯ.**

---

### Шаг 2: Подготовка данных

```bash
mkdir -p data/active
.venv/bin/pip install scikit-learn -q
.venv/bin/python -c "
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet('data/labeled/labeled_final.parquet')
df = df[df['label'] != 'unlabeled'].reset_index(drop=True)

# Фиксированный тест (20%)
train_pool, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Seed (50 примеров, стратифицированно)
seed, pool = train_test_split(train_pool, train_size=50, random_state=42, stratify=train_pool['label'])

seed.to_parquet('data/active/seed.parquet', index=False)
pool.to_parquet('data/active/pool.parquet', index=False)
test.to_parquet('data/active/test.parquet', index=False)

print(f'Seed: {len(seed)} | Pool: {len(pool)} | Test: {len(test)}')
print(f'Seed classes: {seed[\"label\"].value_counts().to_dict()}')
"
```

---

### Шаг 3: AL-цикл по всем трём стратегиям

```bash
# Entropy
.venv/bin/python ~/.claude/skills/smart-sampler/scripts/al_cycle.py \
    --seed data/active/seed.parquet \
    --pool data/active/pool.parquet \
    --test data/active/test.parquet \
    --output data/active/history_entropy.json \
    --strategy entropy \
    --n-iterations 5 \
    --batch-size 20

# Margin
.venv/bin/python ~/.claude/skills/smart-sampler/scripts/al_cycle.py \
    --seed data/active/seed.parquet \
    --pool data/active/pool.parquet \
    --test data/active/test.parquet \
    --output data/active/history_margin.json \
    --strategy margin \
    --n-iterations 5 \
    --batch-size 20

# Random baseline
.venv/bin/python ~/.claude/skills/smart-sampler/scripts/al_cycle.py \
    --seed data/active/seed.parquet \
    --pool data/active/pool.parquet \
    --test data/active/test.parquet \
    --output data/active/history_random.json \
    --strategy random \
    --n-iterations 5 \
    --batch-size 20
```

---

### Шаг 4: Визуализация и сравнение

```bash
.venv/bin/python ~/.claude/skills/smart-sampler/scripts/visualize.py \
    --histories data/active/history_entropy.json data/active/history_margin.json data/active/history_random.json \
    --labels entropy margin random \
    --output data/active/learning_curve.png
```

---

### Шаг 5: Отчёт

Написать `data/active/REPORT.md` самостоятельно на основе результатов:

```markdown
# Active Learning Report

## Настройки
- Стратегии: entropy, margin, random
- Seed: 50, итераций: 5, batch: 20
- Финальный размер обучающей выборки: 150 примеров

## Результаты

| Стратегия | Accuracy (финал) | F1 (финал) | Итераций до X% |
|-----------|-----------------|------------|----------------|
| entropy   | ...             | ...        | ...            |
| margin    | ...             | ...        | ...            |
| random    | ...             | ...        | ...            |

## Вывод

Стратегия entropy достигает X% accuracy уже после N итераций (N*20 примеров),
тогда как random требует N+K итераций. Экономия: K*20 примеров (P%).

## График
![Learning Curves](learning_curve.png)
```

Вставить реальные цифры из history JSON файлов.

---

## Итог шага

- `data/active/seed.parquet` — начальная выборка
- `data/active/history_*.json` — история обучения по стратегиям
- `data/active/learning_curve.png` — сравнение кривых
- `data/active/REPORT.md` — итоговый отчёт

---

## Правила

1. Обязательно сравнить минимум 2 стратегии (entropy vs random)
2. HITL: подтвердить настройки перед запуском
3. Отчёт писать с реальными цифрами, не заглушками
4. Если данных < 100 — предупредить и предложить уменьшить seed/batch
