"""
Поиск датасетов в открытом интернете через DuckDuckGo.
Ищет на: Kaggle, PapersWithCode, GitHub, HuggingFace, UCI, OpenML и в целом по теме.

Использование:
  python search_web.py --topic "product reviews sentiment" --limit 8
  python search_web.py --topic "audio speech emotion" --modality audio --limit 8
"""
import argparse
import sys


SEARCH_TEMPLATES = [
    "{topic} dataset download site:kaggle.com",
    "{topic} dataset site:paperswithcode.com",
    "{topic} dataset site:github.com",
    "{topic} open dataset CSV filetype:csv OR parquet",
    "{topic} dataset machine learning benchmark",
    "{topic} {modality} dataset huggingface",
]


def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    try:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            from ddgs import DDGS
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
                    "snippet": r.get("body", "")[:120],
                })
    except Exception as e:
        print(f"  WARNING: ошибка поиска '{query[:50]}...': {e}")
    return results


def search_web(topic: str, modality: str = "any", limit: int = 8):
    print(f"\n=== Веб-поиск датасетов: '{topic}' (модальность: {modality}) ===\n")

    all_results = []
    seen_urls = set()

    queries = [t.format(topic=topic, modality=modality) for t in SEARCH_TEMPLATES]
    if modality != "any":
        queries.insert(0, f"{topic} {modality} dataset download")

    for query in queries:
        if len(all_results) >= limit:
            break
        results = search_duckduckgo(query, max_results=3)
        for r in results:
            if r["url"] not in seen_urls and len(all_results) < limit:
                seen_urls.add(r["url"])
                all_results.append(r)

    if not all_results:
        print("Ничего не найдено. Попробуй другую тему.")
        return

    # Классифицировать источники
    def classify(url: str) -> str:
        u = url.lower()
        if "kaggle.com" in u:         return "Kaggle"
        if "paperswithcode.com" in u: return "PapersWithCode"
        if "github.com" in u:         return "GitHub"
        if "huggingface.co" in u:     return "HuggingFace"
        if "openml.org" in u:         return "OpenML"
        if "uci.edu" in u:            return "UCI"
        return "Веб"

    print(f"{'#':<3} {'Тип':<15} {'Название':<40} URL")
    print("-" * 100)
    for i, r in enumerate(all_results, 1):
        src = classify(r["url"])
        title = r["title"][:38]
        url_short = r["url"][:55]
        print(f"{i:<3} {src:<15} {title:<40} {url_short}")
        if r["snippet"]:
            print(f"    └─ {r['snippet'][:95]}")

    print(f"\nВсего найдено: {len(all_results)}")
    print("\nПроверь ссылки через WebFetch для подтверждения доступности данных.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск датасетов в интернете через DuckDuckGo")
    parser.add_argument("--topic", required=True, help="Тема для поиска")
    parser.add_argument("--modality", default="any",
                        choices=["any", "text", "image", "audio", "video", "tabular"],
                        help="Модальность данных")
    parser.add_argument("--limit", type=int, default=8, help="Максимум результатов")
    args = parser.parse_args()
    search_web(args.topic, args.modality, args.limit)
