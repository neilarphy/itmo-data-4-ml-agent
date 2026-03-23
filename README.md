# ML Data Pipeline — Claude Code Skills

Универсальный агентный пайплайн для сбора, очистки, разметки и подготовки данных к обучению.

## Концепция

Ты описываешь задачу — агент сам ищет источники, предлагает, ждёт твоего подтверждения и идёт дальше. На каждом ключевом этапе — Human-in-the-loop checkpoint.

```
/ml-pipeline "отзывы на рестораны" --classes "positive,negative,neutral" --task "sentiment classification"
```

```
Dataset Collector → Quality Guard → Auto Tagger → Smart Sampler
       ↓                  ↓               ↓               ↓
 combined.parquet   cleaned.parquet  labeled.parquet   REPORT.md
```

## Скиллы

| Скилл | Что делает | Задание курса |
|-------|------------|---------------|
| `data-collector` | Ищет и собирает данные (HF + парсинг) | Task 1 |
| `quality-guard` | Детектирует и чистит проблемы данных | Task 2 |
| `auto-tagger` | Авторазметка через LLM + HITL review | Task 3 |
| `smart-sampler` | Active Learning: entropy/margin/random | Task 4 |
| `ml-pipeline` | Мета-скилл: запускает всё по порядку | Final |

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

## Human-in-the-loop чекпоинты

| # | Этап | Что проверяешь |
|---|------|----------------|
| 1 | После сбора | Подтверждаешь источники данных |
| 2 | После анализа качества | Выбираешь стратегию чистки |
| 3 | После авторазметки | Редактируешь `review_queue.csv` |
| 4 | Перед обучением | Подтверждаешь финальный датасет |

## Структура данных

```
data/
  raw/           # собранные данные
  eda/           # графики и отчёт EDA
  detective/     # отчёты качества, визуализации
  cleaned/       # очищенный датасет
  labeled/       # размеченный датасет
  active/        # результаты Active Learning
review_queue.csv # HITL: примеры для ручной проверки
```

## Унифицированная схема датасета

| Колонка | Тип | Описание |
|---------|-----|----------|
| `text` | str | Текст |
| `label` | str | Метка класса |
| `source` | str | `hf:<name>` или `scrape:<url>` |
| `collected_at` | str | ISO timestamp |

## Требования

```bash
pip install pandas datasets huggingface_hub requests beautifulsoup4 \
            matplotlib pyarrow openai python-dotenv scikit-learn
```

`.env`:
```
API_KEY=your_key
API_BASE_URL=https://bothub.chat/api/v2/openai/v1
API_MODEL=gpt-4o-mini
```

## Курс

ИТМО — «Сбор и разметка данных для машинного обучения», Весна 2026
