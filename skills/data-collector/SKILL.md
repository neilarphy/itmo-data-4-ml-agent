---
name: data-collector
description: Находит источники данных под ML-задачу (HuggingFace, Kaggle, открытый интернет), скачивает или парсит выбранные пользователем, унифицирует схему, делает EDA и генерирует отчёт. Поддерживает text, image, audio.
license: MIT
metadata:
  category: data-engineering
  version: 1.1.0
---

# Data Collector

Агент для сбора данных под любую ML-задачу. Ищет сразу в нескольких местах, предлагает на выбор, собирает подтверждённое.

## ВАЖНО: Рабочая директория

Все файлы создаются в ТЕКУЩЕЙ рабочей директории (CWD).
Скрипты берутся из `~/.claude/skills/data-collector/scripts/`.
НЕ использовать `source activate` — только `.venv/bin/python`.

---

## Унифицированная схема (ОБЯЗАТЕЛЬНАЯ)

Все источники и модальности приводятся к единому формату:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `text` | str | Текст / путь к файлу / URL медиа |
| `label` | str | Метка класса (`unlabeled` если нет) |
| `modality` | str | `text` / `image` / `audio` |
| `source` | str | `hf:<name>` / `kaggle:<owner/name>` / `scrape:<url>` / `api:<name>` |
| `collected_at` | str | ISO timestamp UTC |

---

## Workflow

### Шаг 0: Setup окружения

```bash
python -m venv .venv
.venv/bin/pip install pandas datasets huggingface_hub requests beautifulsoup4 matplotlib pyarrow kaggle
mkdir -p data/raw data/eda
```

---

### Шаг 1: Поиск источников — три скрипта + парсинг

Запустить ВСЕ три поиска, затем собрать итоговую таблицу.

#### 1.1 HuggingFace Hub

```bash
.venv/bin/pip install huggingface_hub -q
.venv/bin/python ~/.claude/skills/data-collector/scripts/search_datasets.py --source hf --topic "<TOPIC>" --limit 6
```

#### 1.2 Kaggle

```bash
.venv/bin/pip install kaggle -q
.venv/bin/python ~/.claude/skills/data-collector/scripts/search_datasets.py --source kaggle --topic "<TOPIC>" --limit 6
```

#### 1.3 Веб-поиск (Kaggle, PapersWithCode, GitHub, UCI, OpenML и др.)

```bash
.venv/bin/pip install duckduckgo-search -q
.venv/bin/python ~/.claude/skills/data-collector/scripts/search_web.py --topic "<TOPIC>" --modality <text|image|audio|any> --limit 8
```

Скрипт ищет через DuckDuckGo по шаблонам: `site:kaggle.com`, `site:paperswithcode.com`, `site:github.com`, общий поиск датасетов.
Для перспективных ссылок использовать WebFetch чтобы проверить что данные реально доступны.

#### 1.4 Источники для парсинга / API

Самостоятельно поискать сайты с данными по теме:
- Отзывы: Yelp, IMDb, App Store reviews
- Новости: RSS-ленты, GDELT, NewsAPI
- Соцсети: Reddit API (PRAW) — бесплатно без ключа
- Специализированные: зависит от задачи

Для каждого источника оценить:
- URL / endpoint
- Что именно получим (тексты, изображения, аудио)
- Примерный объём
- Нужна ли разметка

---

### Шаг 2: HITL Checkpoint #1 — выбор источников

Показать СВОДНУЮ таблицу по всем найденным источникам:

```
## Найденные источники для "<TOPIC>"

### Готовые датасеты:
| # | Источник | Название | Размер | Модальность | Метки | Примечание |
|---|----------|----------|--------|-------------|-------|------------|
| 1 | HuggingFace | ... | ... | text | есть | ... |
| 2 | Kaggle | ... | ... | image | есть | ... |
| 3 | PapersWithCode | ... | ... | text | есть | ... |

### Для парсинга / API:
| # | URL / API | Что получим | ~Объём | Модальность | Метки |
|---|-----------|-------------|--------|-------------|-------|
| 4 | reddit.com/r/... | комментарии | ~10K | text | нет |
| 5 | ... | ... | ... | ... | ... |

Какие источники берём? Укажи номера (например: 1, 4):
```

**ЖДАТЬ ОТВЕТА ПОЛЬЗОВАТЕЛЯ.**

---

### Шаг 3: Сбор данных

Для каждого выбранного источника:

#### HuggingFace:
```bash
.venv/bin/python -c "
from datasets import load_dataset
import pandas as pd
ds = load_dataset('<NAME>', split='train', trust_remote_code=True)
df = ds.to_pandas()
df.to_parquet('data/raw/<NAME>.parquet', index=False)
print(f'Скачано: {len(df)} строк')
print(f'Колонки: {list(df.columns)}')
print(df.head(3).to_string())
"
```

#### Kaggle:
```bash
.venv/bin/python ~/.claude/skills/data-collector/scripts/search_datasets.py --source kaggle --download "<owner/name>" --output data/raw/
```

Или скачать вручную через kaggle API:
```bash
.venv/bin/python -c "import kaggle; kaggle.api.dataset_download_files('<owner/name>', path='data/raw/', unzip=True)"
```

#### Парсинг сайта:
Написать парсер самостоятельно под конкретный сайт. Обязательно:
- `time.sleep(1-2)` между запросами
- User-Agent header
- Обработка ошибок (try/except + retry)
- `label = 'unlabeled'` для данных без меток
- Сохранить в `data/raw/<sitename>_raw.parquet`

#### API (Reddit, NewsAPI и т.д.):
Написать код под конкретный API. Пример для Reddit (PRAW):
```python
import praw, pandas as pd
# Написать под конкретный subreddit и задачу
```

---

### Шаг 4: Унификация схемы

Для каждого источника:

1. Прочитать и исследовать колонки:
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/raw/<FILE>.parquet')
print('Колонки:', df.columns.tolist())
print('Типы:', df.dtypes.to_string())
print(df.head(3).to_string())
"
```

2. Написать скрипт унификации под конкретный датасет (определить text/label/modality колонки):
```python
import pandas as pd
from datetime import datetime, timezone

df = pd.read_parquet('data/raw/<FILE>.parquet')

unified = pd.DataFrame({
    'text': df['<text_col>'].astype(str),
    'label': df['<label_col>'].astype(str),   # или 'unlabeled'
    'modality': 'text',                         # или 'image', 'audio'
    'source': 'hf:<name>',
    'collected_at': datetime.now(timezone.utc).isoformat()
})

unified = unified[unified['text'].str.strip() != '']
unified.to_parquet('data/raw/<NAME>_unified.parquet', index=False)
print(f'Унифицировано: {len(unified)} строк')
```

---

### Шаг 5: Объединение

```bash
.venv/bin/python -c "
import pandas as pd, glob
files = glob.glob('data/raw/*_unified.parquet')
print(f'Объединяю: {files}')
dfs = [pd.read_parquet(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
combined = combined.drop_duplicates(subset=['text'])
combined.to_parquet('data/raw/combined.parquet', index=False)
print(f'Итого: {len(combined)} строк')
print(combined.groupby(['source','modality']).size().to_string())
"
```

---

### Шаг 6: EDA

Написать EDA-скрипт под конкретные данные. Минимум:

```python
import pandas as pd, matplotlib.pyplot as plt, json
from collections import Counter

df = pd.read_parquet('data/raw/combined.parquet')
modalities = df['modality'].unique()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Распределение классов
df['label'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Распределение классов')

# 2. Источники
df['source'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.0f%%')
axes[1].set_title('Источники')

# 3. Модальность-специфичная метрика
if 'text' in modalities:
    df[df['modality']=='text']['text'].str.len().hist(ax=axes[2], bins=50)
    axes[2].set_title('Длина текстов')
else:
    df['modality'].value_counts().plot(kind='bar', ax=axes[2])
    axes[2].set_title('Модальности')

plt.tight_layout()
plt.savefig('data/eda/eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()

stats = {
    'total_rows': len(df),
    'modalities': df['modality'].value_counts().to_dict(),
    'sources': df['source'].value_counts().to_dict(),
    'labels': df['label'].value_counts().to_dict(),
}
if 'text' in modalities:
    tl = df[df['modality']=='text']['text'].str.len()
    stats.update({'text_len_mean': float(tl.mean()), 'text_len_p95': float(tl.quantile(0.95))})

import json
with open('data/eda/stats.json', 'w') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(json.dumps(stats, ensure_ascii=False, indent=2))
```

---

### Шаг 7: Отчёт

```bash
.venv/bin/python ~/.claude/skills/data-collector/scripts/generate_report.py \
    --stats data/eda/stats.json \
    --output data/eda/REPORT.md
```

---

## Итог шага

- `data/raw/combined.parquet` — объединённый датасет
- `data/eda/eda_overview.png` — визуализации
- `data/eda/stats.json` — статистика
- `data/eda/REPORT.md` — отчёт

---

## Правила

1. Искать источники во ВСЕХ четырёх направлениях (HF, Kaggle, веб, парсинг)
2. ОБЯЗАТЕЛЬНО ждать подтверждения источников у пользователя
3. Парсеры и API-клиенты писать самостоятельно под конкретный сайт
4. EDA писать под конкретные данные и модальность
5. Всегда приводить к унифицированной схеме
6. Все файлы в текущей рабочей директории
