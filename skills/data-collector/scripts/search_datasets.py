"""
Поиск датасетов на HuggingFace и Kaggle.

Использование:
  python search_datasets.py --source hf --topic "product reviews" --limit 6
  python search_datasets.py --source kaggle --topic "sentiment" --limit 6
  python search_datasets.py --source kaggle --download "owner/dataset-name" --output data/raw/
"""
import argparse
import sys
import os


def search_huggingface(topic: str, limit: int = 6):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    try:
        datasets = list(api.list_datasets(search=topic, limit=limit * 3))
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    results = []
    for ds in datasets:
        tags = getattr(ds, "tags", []) or []
        size = next((t.replace("size_categories:", "") for t in tags if t.startswith("size_categories:")), "?")
        task = next((t.replace("task_categories:", "") for t in tags if t.startswith("task_categories:")), "?")
        modality = next((t.replace("modality:", "") for t in tags if t.startswith("modality:")), "text")
        results.append({
            "id": ds.id,
            "size": size,
            "task": task,
            "modality": modality,
            "downloads": getattr(ds, "downloads", 0) or 0,
            "likes": getattr(ds, "likes", 0) or 0,
        })

    results.sort(key=lambda x: x["downloads"], reverse=True)
    results = results[:limit]

    print(f"\n=== HuggingFace: '{topic}' ({len(results)} результатов) ===\n")
    print(f"{'#':<3} {'ID':<38} {'Размер':<12} {'Задача':<22} {'Модальность':<12} {'↓':<10} {'♥'}")
    print("-" * 105)
    for i, ds in enumerate(results, 1):
        print(f"{i:<3} {ds['id']:<38} {ds['size']:<12} {ds['task']:<22} {ds['modality']:<12} {ds['downloads']:<10} {ds['likes']}")

    print(f"\nЗагрузить: from datasets import load_dataset; ds = load_dataset('<ID>', split='train')")


def search_kaggle(topic: str, limit: int = 6):
    try:
        import kaggle
    except ImportError:
        print("ERROR: pip install kaggle  (и настрой ~/.kaggle/kaggle.json)")
        print("\nАльтернатива — поиск вручную: https://www.kaggle.com/datasets?search=" + topic.replace(" ", "+"))
        return

    try:
        datasets = kaggle.api.dataset_list(search=topic, sort_by="votes", page=1)
    except Exception as e:
        print(f"ERROR Kaggle API: {e}")
        print(f"Поищи вручную: https://www.kaggle.com/datasets?search={topic.replace(' ', '+')}")
        return

    results = list(datasets)[:limit]
    if not results:
        print(f"Kaggle: ничего не найдено по '{topic}'")
        return

    print(f"\n=== Kaggle: '{topic}' ({len(results)} результатов) ===\n")
    print(f"{'#':<3} {'Owner/Name':<40} {'Размер':<12} {'Votes':<8} {'Downloads'}")
    print("-" * 80)
    for i, ds in enumerate(results, 1):
        ref = f"{ds.ref}"
        size = f"{ds.totalBytes / 1e6:.1f} MB" if hasattr(ds, "totalBytes") else "?"
        votes = getattr(ds, "voteCount", 0)
        dl = getattr(ds, "downloadCount", 0)
        print(f"{i:<3} {ref:<40} {size:<12} {votes:<8} {dl}")

    print(f"\nСкачать: kaggle datasets download -d <owner/name> -p data/raw/ --unzip")


def download_kaggle(dataset_ref: str, output_dir: str):
    try:
        import kaggle
    except ImportError:
        print("ERROR: pip install kaggle")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Скачиваю {dataset_ref} → {output_dir}")
    kaggle.api.dataset_download_files(dataset_ref, path=output_dir, unzip=True)
    print(f"Готово: {output_dir}")

    # Показать что скачалось
    files = os.listdir(output_dir)
    print(f"Файлы: {files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск датасетов на HF и Kaggle")
    parser.add_argument("--source", choices=["hf", "kaggle"], required=True)
    parser.add_argument("--topic", help="Тема для поиска")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--download", help="Kaggle ref для скачивания: owner/name")
    parser.add_argument("--output", help="Папка для скачивания", default="data/raw/")
    args = parser.parse_args()

    if args.download:
        download_kaggle(args.download, args.output)
    elif args.source == "hf":
        search_huggingface(args.topic, args.limit)
    elif args.source == "kaggle":
        search_kaggle(args.topic, args.limit)
