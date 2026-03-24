---
name: auto-tagger
description: Агент для автоматической разметки данных через LLM API. Анализирует датасет, подтверждает задачу у пользователя, размечает батчами, флагает неуверенные примеры для HITL-проверки, генерирует спецификацию и экспортирует в LabelStudio.
license: MIT
metadata:
  category: data-annotation
  version: 1.0.0
---

# Auto Tagger

Агент для автоматической LLM-разметки данных с Human-in-the-loop.

## ВАЖНО

Все файлы в ТЕКУЩЕЙ рабочей директории.
Вход: `data/cleaned/cleaned.parquet` (результат quality-guard).
Выход: `data/labeled/labeled.parquet` + `data/labeled/review_queue.csv`.

Требуется `.env` с ключом API:
```
API_KEY=...
API_BASE_URL=https://bothub.chat/api/v2/openai/v1
API_MODEL=gpt-4o-mini
```

### Поддержка модальностей

| Модальность | Метод | Требования |
|-------------|-------|------------|
| `text` | Батч из 10 текстов в один LLM запрос | — |
| `image` | GPT-4o-mini vision API, по одному | API_MODEL=gpt-4o-mini (поддерживает vision) |
| `audio` | faster-whisper tiny → транскрипция → LLM | `pip install faster-whisper` (~75MB) |

Модальность определяется автоматически из колонки `modality` датасета.

---

## Workflow

### Шаг 0: Проверка входных данных

```bash
.venv/bin/python -c "
import pandas as pd, os
path = 'data/cleaned/cleaned.parquet'
if not os.path.exists(path):
    print('ERROR: файл не найден. Сначала запусти /quality-guard')
    exit(1)
df = pd.read_parquet(path)
labeled = (df['label'] != 'unlabeled').sum() if 'label' in df.columns else 0
unlabeled = (df['label'] == 'unlabeled').sum() if 'label' in df.columns else len(df)
print(f'Строк: {len(df)}')
print(f'Уже размечено: {labeled}')
print(f'Нужно разметить: {unlabeled}')
print(f'Классы: {df[\"label\"].value_counts().to_dict() if \"label\" in df.columns else \"нет\"}')
print(df.head(3).to_string())
"
```

---

### Шаг 1: HITL Checkpoint #3a — подтверждение задачи разметки

Показать пользователю:

```
## Настройки авторазметки

- Файл: data/cleaned/cleaned.parquet
- Строк для разметки: <N>
- Классы: <список из аргументов или найденные в данных>
- Задача: <описание из аргументов>
- Модель: <API_MODEL из .env>
- Порог уверенности: 0.75 (примеры ниже → review_queue.csv)

Размечаем? [да / нет]:
```

**ЖДАТЬ ОТВЕТА ПОЛЬЗОВАТЕЛЯ.**

---

### Шаг 2: Авторазметка через LLM

```bash
mkdir -p data/labeled
.venv/bin/pip install openai python-dotenv -q
.venv/bin/python ~/.claude/skills/auto-tagger/scripts/auto_labeler.py \
    --input data/cleaned/cleaned.parquet \
    --output data/labeled/labeled.parquet \
    --classes "<class1,class2,...>" \
    --task "<описание задачи>" \
    --confidence-threshold 0.75
```

Скрипт:
- Размечает батчами по 10 текстов
- Для каждого примера возвращает `label` + `confidence` (0.0–1.0)
- Примеры с `confidence < threshold` помечает как `needs_review`
- Сохраняет прогресс каждые 100 строк (можно прервать и продолжить)

---

### Шаг 3: HITL Checkpoint #3b — ручная проверка неуверенных примеров

После разметки скрипт автоматически создаёт `review_queue.csv`.

Показать пользователю:

```
## Авторазметка завершена

- Размечено автоматически: <N> примеров
- Средняя уверенность: <mean_confidence>%
- Флагов на проверку (confidence < 0.75): <review_count> примеров

### Следующий шаг — ОБЯЗАТЕЛЬНО:

Открой файл review_queue.csv и проверь метки.
Исправь колонку `label` там где агент ошибся.
Сохрани файл.

Когда готово — напиши "готово" и я продолжу.
```

**ЖДАТЬ ПОДТВЕРЖДЕНИЯ.** После этого:

```bash
.venv/bin/python -c "
import pandas as pd

labeled = pd.read_parquet('data/labeled/labeled.parquet')
reviewed = pd.read_csv('review_queue.csv')

# Заменить метки из review_queue
high_conf = labeled[labeled['confidence'] >= 0.75]
merged = pd.concat([high_conf, reviewed[['text','label','source','collected_at','modality','confidence']]], ignore_index=True)
merged = merged.drop_duplicates(subset=['text'], keep='last')
merged.to_parquet('data/labeled/labeled_final.parquet', index=False)
print(f'Финальный датасет: {len(merged)} строк')
print(merged['label'].value_counts())
"
```

---

### Шаг 4: Спецификация разметки

Написать `data/labeled/spec.md` самостоятельно на основе задачи и классов:

```markdown
# Annotation Specification — <TASK>

## Задача
<описание ML-задачи>

## Классы
### <class_name>
**Определение:** ...
**Примеры:**
- "текст примера 1"
- "текст примера 2"
- "текст примера 3"
**Граничные случаи:** ...

[повторить для каждого класса]

## Инструкция для разметчика
...

## Граничные случаи
...
```

Взять 3+ реальных примера из датасета для каждого класса.

---

### Шаг 5: Метрики качества

```bash
.venv/bin/python ~/.claude/skills/auto-tagger/scripts/quality_check.py \
    --input data/labeled/labeled_final.parquet \
    --output data/labeled/quality.json
```

---

### Шаг 6: Экспорт в LabelStudio

```bash
.venv/bin/python ~/.claude/skills/auto-tagger/scripts/export_labelstudio.py \
    --input data/labeled/labeled_final.parquet \
    --output data/labeled/labelstudio_import.json \
    --task "<TASK>" \
    --classes "<class1,class2,...>"
```

---

## Итог шага

- `data/labeled/labeled_final.parquet` — размеченный датасет (авто + проверенный)
- `review_queue.csv` — примеры прошедшие ручную проверку
- `data/labeled/spec.md` — спецификация разметки
- `data/labeled/quality.json` — метрики качества
- `data/labeled/labelstudio_import.json` — экспорт для LabelStudio

---

## Правила

1. ОБЯЗАТЕЛЬНО подтверждать задачу разметки у пользователя (Шаг 1)
2. ОБЯЗАТЕЛЬНО создавать review_queue.csv и ждать проверки (Шаг 3)
3. Спецификацию писать с реальными примерами из датасета
4. Не пропускать экспорт в LabelStudio
5. Если API недоступен — сообщить и предложить альтернативу (zero-shot через HF transformers)
