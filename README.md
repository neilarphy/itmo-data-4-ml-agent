# ML Data Pipeline — Claude Code Skills

Агентный пайплайн для сбора, очистки, разметки и подготовки данных к обучению.
Поддерживает модальности: **text, image, audio, tabular**.

## Концепция

Описываешь задачу — агент сам ищет источники, предлагает, ждёт подтверждения и идёт дальше.
На каждом ключевом этапе — Human-in-the-loop checkpoint.

```
/ml-pipeline "cow behavior" --classes "moving,resting,ruminating" --task "IMU-based activity classification"
```

```
Data Collector → Quality Guard → Auto Tagger → Smart Sampler
      ↓                ↓               ↓               ↓
combined.parquet  cleaned.parquet  labeled.parquet  REPORT.md + model.pkl
```

## Скиллы

| Скилл | Что делает | Задание курса |
|-------|------------|---------------|
| `data-collector` | Ищет данные на HF / Kaggle / Roboflow / веб, скачивает, делает EDA | Task 1 |
| `quality-guard` | Аудит: пропуски, дубли, выбросы, дисбаланс; чистка по стратегии | Task 2 |
| `auto-tagger` | Авторазметка через LLM (text/image/audio) + HITL review_queue | Task 3 |
| `smart-sampler` | Active Learning: entropy / margin / random, learning curves | Task 4 |
| `ml-pipeline` | Мета-скилл: запускает все 4 этапа по порядку | Final |

## Установка

```bash
bash install_skills.sh
```

Копирует все скиллы в `~/.claude/skills/`.

## Запуск

### Полный пайплайн
```
/ml-pipeline "<тема>" --classes "<классы>" --task "<описание задачи>"
```

### Отдельные скиллы
```
/data-collector "<тема>"
/quality-guard
/auto-tagger --classes "<классы>" --task "<описание>"
/smart-sampler
```

## Скрипты

### data-collector
| Скрипт | Описание |
|--------|----------|
| `search_datasets.py` | Поиск датасетов на HuggingFace и Kaggle |
| `search_web.py` | DuckDuckGo-поиск + парсинг сайтов |
| `search_roboflow.py` | Поиск и скачивание image-датасетов с Roboflow Universe |
| `generate_report.py` | EDA: распределения, примеры, сводка → `data/eda/REPORT.md` |

### quality-guard
| Скрипт | Описание |
|--------|----------|
| `audit.py` | Аудит: пропуски / дубли / выбросы (IQR+z) / дисбаланс → `problems.json` + PNG |
| `cleaner.py` | Чистка по стратегии: `aggressive` / `conservative` / `balanced` |
| `reporter.py` | Сравнительный отчёт до/после → `comparison.md` |

### auto-tagger
| Скрипт | Описание |
|--------|----------|
| `auto_labeler.py` | LLM-разметка: text (chat), image (GPT-4o-mini vision, base64), audio (faster-whisper → LLM) |
| `quality_check.py` | Метрики разметки: уверенность, покрытие, распределение |
| `export_labelstudio.py` | Экспорт в формат Label Studio |

### smart-sampler
| Скрипт | Описание |
|--------|----------|
| `al_cycle.py` | AL-цикл: entropy / margin / random; мультимодальность: TF-IDF / sentence-transformers / ResNet18 (512-dim) / MFCC+delta (~88-dim) / tabular (JSON→StandardScaler); модели: logreg / svm / rf |
| `visualize.py` | Кривые обучения — сравнение стратегий → `learning_curve.png` |
| `predict.py` | Inference на сохранённой модели (все модальности) |

### ml-pipeline
| Скрипт | Описание |
|--------|----------|
| `generate_datacard.py` | Авто-генерация `DATA_CARD.md` из артефактов пайплайна |

## Human-in-the-loop чекпоинты

| # | Этап | Что проверяешь |
|---|------|----------------|
| 1 | После поиска источников | Подтверждаешь какие датасеты скачивать |
| 2 | После аудита качества | Выбираешь стратегию чистки |
| 3 | После авторазметки | Редактируешь `review_queue.csv` — исправляешь ошибочные метки |
| 4 | Перед AL | Подтверждаешь настройки seed / итераций / batch |

## Структура

```
data/
  raw/            # собранные сырые данные
  eda/            # EDA графики и REPORT.md
  detective/      # аудит качества: problems.json, PNG, comparison.md
  cleaned/        # очищенный датасет
  labeled/        # размеченный датасет + spec.md + labelstudio_import.json
  active/         # AL: history_*.json, learning_curve.png, REPORT.md
models/
  final_model.pkl # обученная модель (vectorizer + classifier)
review_queue.csv  # HITL: примеры для ручной проверки и правки
DATA_CARD.md      # описание датасета
```

## Унифицированная схема датасета

| Колонка | Тип | Описание |
|---------|-----|----------|
| `text` | str | Текст / путь к файлу / JSON с числовыми фичами (tabular) |
| `label` | str | Метка класса |
| `modality` | str | `text` / `image` / `audio` / `tabular` |
| `source` | str | `hf:<name>`, `kaggle:<name>`, `scrape:<url>`, `roboflow:<name>` |
| `collected_at` | str | ISO timestamp |

## Требования

```bash
pip install pandas datasets huggingface_hub requests beautifulsoup4 \
            matplotlib pyarrow openai python-dotenv scikit-learn \
            duckduckgo-search Pillow
# Опционально:
pip install torch torchvision          # ResNet18 image features
pip install sentence-transformers      # семантические эмбеддинги для текста
pip install librosa                    # MFCC audio features
pip install faster-whisper             # транскрипция аудио
pip install roboflow                   # скачивание с Roboflow Universe
```

`.env`:
```
API_KEY=your_openai_compatible_key
API_BASE_URL=https://bothub.chat/api/v2/openai/v1
API_MODEL=gpt-4o-mini
ROBOFLOW_API_KEY=your_roboflow_key   # опционально, для search_roboflow.py
```

## Курс

ИТМО — «Сбор и разметка данных для машинного обучения», Весна 2026
