---
name: ml-pipeline
description: Мета-скилл. Запускает полный пайплайн подготовки ML-датасета одной командой. Оркестрирует data-collector → quality-guard → auto-tagger → smart-sampler. На каждом этапе — Human-in-the-loop checkpoint. Идёт сам, не останавливается без необходимости.
license: MIT
metadata:
  category: ml-engineering
  version: 1.0.0
---

# ML Pipeline

Полный пайплайн от сырых данных до обученной модели с Active Learning.

## Запуск

```
/ml-pipeline "<TOPIC>" --classes "<class1,class2,...>" --task "<описание задачи ML>"
```

Пример:
```
/ml-pipeline "restaurant reviews" --classes "positive,negative,neutral" --task "Sentiment classification of restaurant reviews"
```

---

## Аргументы

| Аргумент | Обязательный | Описание |
|----------|-------------|----------|
| `<TOPIC>` | Да | Тема для поиска данных |
| `--classes` | Да | Классы через запятую |
| `--task` | Да | Описание задачи классификации |
| `--strategy` | Нет (balanced) | Стратегия чистки: aggressive/conservative/balanced |
| `--confidence` | Нет (0.75) | Порог уверенности авторазметки |
| `--seed` | Нет (50) | Начальный размер seed для AL |
| `--iterations` | Нет (5) | Итераций Active Learning |
| `--batch` | Нет (20) | Batch size для AL |

---

## Поток данных

```
[1] data-collector  →  data/raw/combined.parquet
         ↓
[2] quality-guard   →  data/cleaned/cleaned.parquet
         ↓
[3] auto-tagger     →  data/labeled/labeled_final.parquet  +  review_queue.csv
         ↓
[4] smart-sampler   →  data/active/learning_curve.png  +  data/active/REPORT.md
```

---

## Workflow

### Этап 0: Setup

```bash
python -m venv .venv
.venv/bin/pip install pandas datasets huggingface_hub requests beautifulsoup4 \
    matplotlib pyarrow kaggle duckduckgo-search openai python-dotenv scikit-learn
mkdir -p data/raw data/eda data/detective data/cleaned data/labeled data/active
```

Проверить `.env`:
```bash
.venv/bin/python -c "
from dotenv import load_dotenv; import os; load_dotenv()
key = os.getenv('API_KEY')
print('API_KEY:', '✓ найден' if key else '✗ НЕ НАЙДЕН — авторазметка не заработает')
"
```

Уточнить у пользователя если что-то не так.

---

### Этап 1: Data Collector

**Цель:** собрать данные из 2+ источников (HF/Kaggle/веб + парсинг/API).

**Действия:**
1. Запустить поиск по всем трём скриптам:
```bash
.venv/bin/python ~/.claude/skills/data-collector/scripts/search_datasets.py --source hf --topic "<TOPIC>" --limit 6
.venv/bin/python ~/.claude/skills/data-collector/scripts/search_datasets.py --source kaggle --topic "<TOPIC>" --limit 6
.venv/bin/python ~/.claude/skills/data-collector/scripts/search_web.py --topic "<TOPIC>" --limit 8
```

2. Найти сайты/API для парсинга по теме.

**HUMAN CHECKPOINT #1:**
```
## Этап 1/4: Data Collector

Найденные источники для "<TOPIC>":

### Готовые датасеты:
| # | Источник | Название | Размер | Модальность |
|---|----------|----------|--------|-------------|
...

### Для парсинга / API:
| # | URL/API | Что получим | ~Объём | Метки |
...

Какие берём? Укажи номера (минимум 2, один из каждой группы):
```

После подтверждения:
- Скачать / спарсить выбранные источники
- Написать скрипты унификации под каждый
- Объединить в `data/raw/combined.parquet`
- Сделать EDA и создать `data/eda/REPORT.md`

Сообщить: `✓ Этап 1 завершён: <N> строк, <M> источников → data/raw/combined.parquet`

---

### Этап 2: Quality Guard

**Цель:** найти и устранить проблемы качества.

**Действия:**
```bash
mkdir -p data/detective
.venv/bin/python ~/.claude/skills/quality-guard/scripts/audit.py \
    --input data/raw/combined.parquet --output data/detective
```

**HUMAN CHECKPOINT #2:**
```
## Этап 2/4: Quality Guard

| Проблема | Кол-во | % | Серьёзность |
...

Стратегии: aggressive / conservative / balanced

Какую стратегию применить? [balanced]:
```

После подтверждения:
```bash
mkdir -p data/cleaned
.venv/bin/python ~/.claude/skills/quality-guard/scripts/cleaner.py \
    --input data/raw/combined.parquet \
    --output data/cleaned/cleaned.parquet \
    --strategy <STRATEGY>

.venv/bin/python ~/.claude/skills/quality-guard/scripts/reporter.py \
    --before data/raw/combined.parquet \
    --after data/cleaned/cleaned.parquet \
    --problems data/detective/problems.json \
    --strategy <STRATEGY> \
    --output data/detective/comparison.md
```

Сообщить: `✓ Этап 2 завершён: <N> строк после чистки → data/cleaned/cleaned.parquet`

---

### Этап 3: Auto Tagger

**Цель:** автоматически разметить данные с LLM, флагнуть неуверенные на проверку.

**Действия:**

**HUMAN CHECKPOINT #3a:**
```
## Этап 3/4: Auto Tagger

- Строк для разметки: <N>
- Классы: <CLASSES>
- Задача: <TASK>
- Порог уверенности: <CONFIDENCE>
- Примеры с confidence < порога → review_queue.csv (ручная проверка)

Размечаем? [да]:
```

После подтверждения:
```bash
mkdir -p data/labeled
.venv/bin/python ~/.claude/skills/auto-tagger/scripts/auto_labeler.py \
    --input data/cleaned/cleaned.parquet \
    --output data/labeled/labeled.parquet \
    --classes "<CLASSES>" \
    --task "<TASK>" \
    --confidence-threshold <CONFIDENCE>
```

**HUMAN CHECKPOINT #3b — РЕАЛЬНАЯ ПРАВКА МЕТОК:**
```
## Авторазметка завершена

- Размечено: <N> примеров
- Средняя уверенность: <X>%
- На проверку: <K> примеров → review_queue.csv

❗ ДЕЙСТВИЕ ТРЕБУЕТСЯ:
1. Открой файл review_queue.csv
2. Проверь колонку 'label' — исправь ошибки в 'corrected_label'
3. Сохрани файл
4. Напиши "готово"
```

**ЖДАТЬ "готово" от пользователя.**

После:
```bash
.venv/bin/python -c "
import pandas as pd

labeled = pd.read_parquet('data/labeled/labeled.parquet')
reviewed = pd.read_csv('review_queue.csv')

# Применить исправления из review_queue
reviewed['label'] = reviewed['corrected_label']
high_conf = labeled[labeled['confidence'] >= 0.75]
result = pd.concat([high_conf, reviewed[['text','label','source','collected_at','modality','confidence']]], ignore_index=True)
result = result.drop_duplicates(subset=['text'], keep='last')
result.to_parquet('data/labeled/labeled_final.parquet', index=False)
print(f'Финальный датасет: {len(result)} строк')
print(result['label'].value_counts().to_string())
"

.venv/bin/python ~/.claude/skills/auto-tagger/scripts/quality_check.py \
    --input data/labeled/labeled_final.parquet \
    --output data/labeled/quality.json

.venv/bin/python ~/.claude/skills/auto-tagger/scripts/export_labelstudio.py \
    --input data/labeled/labeled_final.parquet \
    --output data/labeled/labelstudio_import.json \
    --task "<TASK>" \
    --classes "<CLASSES>"
```

Написать `data/labeled/spec.md` — спецификацию разметки с реальными примерами.

Сообщить: `✓ Этап 3 завершён: <N> размечено, <K> проверено вручную → data/labeled/labeled_final.parquet`

---

### Этап 4: Smart Sampler (Active Learning)

**Цель:** показать, что AL экономит примеры vs случайного выбора.

**Действия:**

**HUMAN CHECKPOINT #4:**
```
## Этап 4/4: Smart Sampler

- Датасет: data/labeled/labeled_final.parquet (<N> строк)
- Seed: <SEED> примеров
- Итераций: <ITERATIONS> × batch <BATCH>
- Стратегии: entropy, margin, random

Подтверждаешь? [да]:
```

После подтверждения:
```bash
mkdir -p data/active

.venv/bin/python -c "
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet('data/labeled/labeled_final.parquet')
df = df[df['label'] != 'unlabeled'].reset_index(drop=True)
train_pool, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
seed, pool = train_test_split(train_pool, train_size=<SEED>, random_state=42, stratify=train_pool['label'])
seed.to_parquet('data/active/seed.parquet', index=False)
pool.to_parquet('data/active/pool.parquet', index=False)
test.to_parquet('data/active/test.parquet', index=False)
print(f'Seed: {len(seed)} | Pool: {len(pool)} | Test: {len(test)}')
"

for strategy in entropy margin random; do
  .venv/bin/python ~/.claude/skills/smart-sampler/scripts/al_cycle.py \
    --seed data/active/seed.parquet \
    --pool data/active/pool.parquet \
    --test data/active/test.parquet \
    --output data/active/history_${strategy}.json \
    --strategy $strategy \
    --n-iterations <ITERATIONS> \
    --batch-size <BATCH>
done

.venv/bin/python ~/.claude/skills/smart-sampler/scripts/visualize.py \
    --histories data/active/history_entropy.json data/active/history_margin.json data/active/history_random.json \
    --labels entropy margin random \
    --output data/active/learning_curve.png
```

Написать `data/active/REPORT.md` с реальными цифрами из history JSON.

Сообщить: `✓ Этап 4 завершён → data/active/learning_curve.png`

---

### Финальная сводка

После завершения всех этапов показать:

```
## ✓ Пайплайн завершён

### Этап 1: Data Collector
- Источников: <N>
- Строк: <N>
- data/raw/combined.parquet | data/eda/REPORT.md

### Этап 2: Quality Guard
- Стратегия: <STRATEGY>
- Строк после чистки: <N>
- data/cleaned/cleaned.parquet | data/detective/comparison.md

### Этап 3: Auto Tagger
- Размечено автоматически: <N>
- Проверено вручную: <K>
- Средняя уверенность: <X>%
- data/labeled/labeled_final.parquet | data/labeled/spec.md

### Этап 4: Smart Sampler
- Лучшая стратегия: <entropy/margin>
- Экономия vs random: <N> примеров (<P>%)
- data/active/learning_curve.png | data/active/REPORT.md

### Готово для ML:
- Размеченный датасет: data/labeled/labeled_final.parquet
- Спецификация: data/labeled/spec.md
- LabelStudio: data/labeled/labelstudio_import.json
```

---

## Правила

1. Идти строго по этапам: 1 → 2 → 3 → 4
2. На каждом HUMAN CHECKPOINT — ждать ответа пользователя
3. Checkpoint #3b — реальная пауза, не просто уведомление
4. Уточнять у пользователя до старта если тема/классы/задача не указаны
5. При ошибке — показать проблему и предложить решение, не прерывать пайплайн
6. Все файлы в ТЕКУЩЕЙ рабочей директории
7. НЕ использовать `source activate` — только `.venv/bin/python`
