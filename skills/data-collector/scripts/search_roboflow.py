"""
Поиск датасетов на Roboflow Universe через DuckDuckGo.
Скачивание через пакет roboflow (требует бесплатный API ключ: app.roboflow.com).

Использование:
  python search_roboflow.py --topic "cow lameness" --limit 8
  python search_roboflow.py --topic "cattle detection" --task object-detection --limit 10

  # Скачать датасет:
  python search_roboflow.py --download --workspace "farm-ai" --project "cow-lameness" --version 1 --format yolov8
"""
import argparse
import os
import re
import sys


def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("ERROR: pip install duckduckgo-search")
        sys.exit(1)

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[:150],
                })
    except Exception as e:
        print(f"  WARNING: ошибка поиска: {e}")
    return results


def parse_roboflow_url(url: str) -> dict | None:
    """Извлечь workspace/project из URL universe.roboflow.com/<ws>/<proj>"""
    m = re.match(r"https?://universe\.roboflow\.com/([^/]+)/([^/?#]+)", url)
    if m:
        return {"workspace": m.group(1), "project": m.group(2)}
    return None


def search_roboflow(topic: str, task: str = "any", limit: int = 8):
    print(f"\n=== Roboflow Universe: '{topic}' (задача: {task}) ===\n")

    queries = [
        f"site:universe.roboflow.com {topic}",
        f"site:universe.roboflow.com {topic} dataset",
    ]
    if task != "any":
        queries.insert(0, f"site:universe.roboflow.com {topic} {task}")

    all_results = []
    seen_urls = set()

    for query in queries:
        if len(all_results) >= limit:
            break
        for r in search_duckduckgo(query, max_results=5):
            url = r["url"]
            if "universe.roboflow.com" not in url:
                continue
            parsed = parse_roboflow_url(url)
            if not parsed:
                continue
            key = (parsed["workspace"], parsed["project"])
            if key in seen_urls:
                continue
            seen_urls.add(key)
            all_results.append({**r, **parsed})
            if len(all_results) >= limit:
                break

    if not all_results:
        print("Ничего не найдено на Roboflow Universe. Попробуй другой запрос.")
        print("Совет: зайди вручную на https://universe.roboflow.com и поищи по теме.")
        return []

    print(f"{'#':<3} {'Workspace/Project':<45} {'Описание'}")
    print("-" * 100)
    for i, r in enumerate(all_results, 1):
        proj = f"{r['workspace']}/{r['project']}"
        title = r["title"][:42]
        print(f"{i:<3} {proj:<45} {title}")
        if r["snippet"]:
            print(f"    └─ {r['snippet'][:90]}")
        print(f"    🔗 {r['url']}")

    print(f"\nВсего найдено: {len(all_results)}")
    print("\n--- Как скачать ---")
    print("1. Получи бесплатный API ключ: https://app.roboflow.com/ → Settings → Roboflow API")
    print("2. Добавь в .env: ROBOFLOW_API_KEY=<ключ>")
    print("3. Запусти скачивание:")
    print(f"   python search_roboflow.py --download --workspace <ws> --project <proj> --version 1 --format folder")
    print("\nДоступные форматы: folder (raw), yolov8, coco, voc, tensorflow, createml")

    return all_results


def download_roboflow(workspace: str, project: str, version: int,
                      fmt: str, output_dir: str):
    """Скачать датасет с Roboflow Universe."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: pip install roboflow")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY не найден в .env")
        print("Получи ключ на https://app.roboflow.com/ → Settings → Roboflow API")
        sys.exit(1)

    print(f"\nСкачиваю {workspace}/{project} v{version} (формат: {fmt})...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(fmt, location=output_dir, overwrite=True)
    print(f"Сохранено в: {output_dir}")

    # Создать parquet из скачанного датасета (для image датасетов)
    _convert_to_parquet(output_dir, workspace, project)


def _convert_to_parquet(dataset_dir: str, workspace: str, project: str):
    """Конвертировать скачанный image датасет в unified parquet схему."""
    import glob as glob_mod
    from datetime import datetime, timezone
    import pandas as pd

    rows = []
    collected_at = datetime.now(timezone.utc).isoformat()

    # Ищем изображения в train/valid/test
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_dir, split, "images")
        if not os.path.isdir(img_dir):
            # некоторые форматы кладут прямо в split/
            img_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(img_dir):
            continue

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            for img_path in glob_mod.glob(os.path.join(img_dir, ext)):
                rows.append({
                    "text": os.path.abspath(img_path),
                    "label": "unlabeled",
                    "modality": "image",
                    "source": f"roboflow:{workspace}/{project}",
                    "collected_at": collected_at,
                })

    if not rows:
        print("WARNING: изображения не найдены, parquet не создан.")
        return

    df = pd.DataFrame(rows)
    out_path = os.path.join("data", "raw", f"roboflow_{project}.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nUnified parquet: {out_path} ({len(df)} изображений)")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск и скачивание датасетов Roboflow Universe")
    parser.add_argument("--topic", help="Тема для поиска")
    parser.add_argument("--task", default="any",
                        choices=["any", "object-detection", "classification", "segmentation", "keypoint"],
                        help="Тип задачи CV")
    parser.add_argument("--limit", type=int, default=8, help="Максимум результатов поиска")

    parser.add_argument("--download", action="store_true", help="Режим скачивания датасета")
    parser.add_argument("--workspace", help="Workspace на Roboflow")
    parser.add_argument("--project", help="Название проекта")
    parser.add_argument("--version", type=int, default=1, help="Версия датасета")
    parser.add_argument("--format", default="folder",
                        choices=["folder", "yolov8", "coco", "voc", "tensorflow", "createml"],
                        help="Формат скачивания")
    parser.add_argument("--output", default="data/raw/roboflow", help="Куда сохранить")

    args = parser.parse_args()

    if args.download:
        if not args.workspace or not args.project:
            print("ERROR: --workspace и --project обязательны для --download")
            sys.exit(1)
        download_roboflow(args.workspace, args.project, args.version, args.format, args.output)
    elif args.topic:
        search_roboflow(args.topic, args.task, args.limit)
    else:
        parser.print_help()
