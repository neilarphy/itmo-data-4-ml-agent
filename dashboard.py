"""
ML Pipeline Dashboard — Streamlit
Запуск: streamlit run dashboard.py
"""
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ML Pipeline Dashboard", page_icon="🔬", layout="wide")

# ── Пути ──────────────────────────────────────────────────────────────────────
PATHS = {
    "raw":      "data/raw/combined.parquet",
    "cleaned":  "data/cleaned/cleaned.parquet",
    "labeled":  "data/labeled/labeled_final.parquet",
    "problems": "data/detective/problems.json",
    "eda":      "data/eda/stats.json",
    "al_entropy": "data/active/history_entropy.json",
    "al_margin":  "data/active/history_margin.json",
    "al_random":  "data/active/history_random.json",
    "review":   "review_queue.csv",
    "model":    "models/final_model.pkl",
    "detective_dir": "data/detective",
}

# ── Хелперы ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_parquet(p):
    return pd.read_parquet(p) if os.path.exists(p) else None

@st.cache_data
def load_json(p):
    if not os.path.exists(p): return None
    with open(p, encoding="utf-8") as f: return json.load(f)

def load_review():
    if not os.path.exists(PATHS["review"]): return None
    return pd.read_csv(PATHS["review"])

def text_col(df):
    """Находит колонку с текстом/превью контента."""
    for c in ("text", "text_preview", "content", "data"):
        if c in df.columns: return c
    return df.columns[0]

def exists(p): return os.path.exists(p)
def icon(p):   return "✅" if exists(p) else "⬜"

# ── Сайдбар ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 ML Pipeline")
    page = st.radio("", ["📊 Обзор", "✏️ HITL Разметка", "📈 Метрики", "💡 Выводы"])
    st.divider()
    st.caption("streamlit run dashboard.py")

# ══════════════════════════════════════════════════════════════════════════════
# ОБЗОР
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Обзор":
    st.title("📊 Статус пайплайна")

    # Этапы
    stages = [
        ("01 Сбор",      PATHS["raw"],      "combined.parquet"),
        ("02 Чистка",    PATHS["cleaned"],  "cleaned.parquet"),
        ("03 Разметка",  PATHS["labeled"],  "labeled_final.parquet"),
        ("04 AL",        PATHS["al_entropy"],"history_*.json"),
        ("05 Модель",    PATHS["model"],    "final_model.pkl"),
    ]
    cols = st.columns(5)
    for i, (name, path, artifact) in enumerate(stages):
        with cols[i]:
            st.markdown(f"### {icon(path)} {name}")
            st.caption(artifact)

    st.divider()

    # Метрики по этапам
    raw_df     = load_parquet(PATHS["raw"])
    cleaned_df = load_parquet(PATHS["cleaned"])
    labeled_df = load_parquet(PATHS["labeled"])

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("📦 Сырые данные")
        if raw_df is not None:
            st.metric("Примеров", f"{len(raw_df):,}")
            if "modality" in raw_df.columns:
                st.dataframe(raw_df["modality"].value_counts().rename("кол-во"), use_container_width=True)
            if "source" in raw_df.columns:
                sources = raw_df["source"].str.split(":").str[0].value_counts()
                st.dataframe(sources.rename("источников"), use_container_width=True)
        else:
            st.info("Нет данных")

    with c2:
        st.subheader("🧹 После чистки")
        if cleaned_df is not None:
            st.metric("Примеров", f"{len(cleaned_df):,}")
            if raw_df is not None:
                dropped = len(raw_df) - len(cleaned_df)
                pct = dropped / len(raw_df) * 100
                st.metric("Удалено", f"{dropped:,}", delta=f"-{pct:.1f}%", delta_color="inverse")
            problems = load_json(PATHS["problems"])
            if problems:
                st.metric("Дубликаты",  problems.get("duplicate_count", "—"))
                st.metric("Выбросы",    problems.get("outlier_count", "—"))
                st.metric("Дисбаланс",  f"{problems.get('imbalance_ratio', 0):.1f}×")
        else:
            st.info("Нет данных")

    with c3:
        st.subheader("🏷️ Размеченные")
        if labeled_df is not None:
            st.metric("Примеров", f"{len(labeled_df):,}")
            if "label" in labeled_df.columns:
                st.dataframe(labeled_df["label"].value_counts().rename("кол-во"), use_container_width=True)
            if "confidence" in labeled_df.columns:
                avg = labeled_df["confidence"].mean()
                low = (labeled_df["confidence"] < 0.7).sum()
                st.metric("Ср. уверенность", f"{avg:.2f}")
                st.metric("На HITL (conf<0.7)", f"{low:,}")
        else:
            st.info("Нет данных")

    # AL + модель
    st.divider()
    c4, c5 = st.columns(2)

    with c4:
        st.subheader("📈 Active Learning")
        best_val, best_name = 0, "—"
        for name, path in [("entropy", PATHS["al_entropy"]),
                           ("margin",  PATHS["al_margin"]),
                           ("random",  PATHS["al_random"])]:
            h = load_json(path)
            if h:
                last = h[-1]
                acc = last.get("accuracy", last.get("f1", 0))
                st.metric(name, f"{acc:.3f}")
                if acc > best_val:
                    best_val, best_name = acc, name
        if best_name != "—":
            st.success(f"Лучшая стратегия: **{best_name}** ({best_val:.3f})")

    with c5:
        st.subheader("🤖 Модель")
        if exists(PATHS["model"]):
            st.success("Модель обучена: `models/final_model.pkl`")
            # Итоговые метрики из истории лучшей стратегии
            h = load_json(PATHS["al_entropy"]) or load_json(PATHS["al_margin"])
            if h:
                last = h[-1]
                for k in ("accuracy", "f1", "n_labeled"):
                    if k in last:
                        st.metric(k, f"{last[k]:.3f}" if isinstance(last[k], float) else last[k])
        else:
            st.info("Модель ещё не обучена")

# ══════════════════════════════════════════════════════════════════════════════
# HITL РАЗМЕТКА
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✏️ HITL Разметка":
    st.title("✏️ Human-in-the-Loop: проверка меток")

    df = load_review()
    if df is None:
        st.info("Файл review_queue.csv не найден. Запусти auto-tagger.")
        st.stop()

    if "corrected_label" not in df.columns:
        df["corrected_label"] = df.get("label", df.get("predicted_label", ""))
        df.to_csv(PATHS["review"], index=False, encoding="utf-8-sig")

    # Определяем нужные колонки гибко
    tcol = text_col(df)
    label_col = "label" if "label" in df.columns else (
                "predicted_label" if "predicted_label" in df.columns else df.columns[0])

    # Классы
    all_labels = df[label_col].dropna().unique().tolist()
    classes = sorted(set(str(l) for l in all_labels if str(l) not in ("unlabeled", "nan")))

    # Прогресс
    is_done = df["corrected_label"].notna() & (df["corrected_label"] != "") & (df["corrected_label"] != df[label_col])
    total   = len(df)
    done    = int(is_done.sum())
    pending_df = df[~is_done]

    show_only_pending = st.checkbox("Показывать только непроверенные", value=True)
    view = pending_df if show_only_pending else df

    p1, p2, p3 = st.columns(3)
    p1.metric("Всего", total)
    p2.metric("Исправлено", done)
    p3.metric("Осталось", len(pending_df))
    st.progress(done / total if total > 0 else 0)

    if view.empty:
        st.success("Все примеры проверены! 🎉")
        st.stop()

    st.divider()

    # Навигация
    if "hitl_idx" not in st.session_state:
        st.session_state.hitl_idx = 0
    idx = min(st.session_state.hitl_idx, len(view) - 1)
    row = view.iloc[idx]
    real_idx = view.index[idx]

    modality = str(row.get("modality", "text")) if "modality" in row.index else "text"
    content  = str(row.get(tcol, ""))
    auto_label = str(row.get(label_col, ""))
    pred_label = str(row.get("predicted_label", auto_label)) if "predicted_label" in row.index else auto_label
    confidence = float(row.get("confidence", 0.0))

    # Заголовок примера
    col_info, col_num = st.columns([4, 1])
    with col_info:
        st.markdown(
            f"**Модальность:** `{modality}` &nbsp;|&nbsp; "
            f"**Авто-метка:** `{auto_label}` &nbsp;|&nbsp; "
            f"**Предсказание модели:** `{pred_label}` &nbsp;|&nbsp; "
            f"**Уверенность:** `{confidence:.2f}`"
        )
    with col_num:
        st.caption(f"Пример {idx + 1} / {len(view)}")

    # Контент примера
    if modality == "image" and os.path.exists(content):
        st.image(content, width=500)
    elif modality == "audio" and os.path.exists(content):
        st.audio(content)
    else:
        # Показываем содержимое крупно и читаемо
        display = content[:3000] if content else "⚠️ Нет содержимого"
        st.markdown("**Содержимое примера:**")
        st.code(display, language=None)

    # Показать все поля строки (для отладки / полного контекста)
    with st.expander("Все поля строки"):
        st.json({k: str(v) for k, v in row.items()})

    st.divider()

    # Выбор метки
    current_corrected = str(row.get("corrected_label", auto_label))
    try:
        default_idx = classes.index(current_corrected)
    except ValueError:
        default_idx = classes.index(auto_label) if auto_label in classes else 0

    chosen = st.radio("Правильная метка:", classes, index=default_idx, horizontal=True,
                      key=f"label_{real_idx}")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 4])
    with c1:
        if st.button("✅ Сохранить", type="primary"):
            df.at[real_idx, "corrected_label"] = chosen
            df.to_csv(PATHS["review"], index=False, encoding="utf-8-sig")
            st.session_state.hitl_idx = min(idx + 1, len(view) - 1)
            st.rerun()
    with c2:
        if st.button("⏭️ Пропустить"):
            st.session_state.hitl_idx = min(idx + 1, len(view) - 1)
            st.rerun()
    with c3:
        if st.button("← Назад") and idx > 0:
            st.session_state.hitl_idx -= 1
            st.rerun()

    st.divider()
    with st.expander("Все записи очереди"):
        st.dataframe(df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Метрики":
    st.title("📈 Метрики пайплайна")

    tab1, tab2, tab3, tab4 = st.tabs(["🧹 Качество данных", "🏷️ Аннотации", "📈 Active Learning", "🤖 Модель"])

    # Качество данных
    with tab1:
        problems = load_json(PATHS["problems"])
        if problems:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Пропуски",    problems.get("missing_count", 0))
            c2.metric("Дубликаты",   problems.get("duplicate_count", 0))
            c3.metric("Выбросы",     problems.get("outlier_count", 0))
            imb = problems.get("imbalance_ratio")
            c4.metric("Дисбаланс",   f"{imb:.1f}×" if imb else "—")

            if exists(PATHS["detective_dir"]):
                pngs = sorted(f for f in os.listdir(PATHS["detective_dir"]) if f.endswith(".png"))
                if pngs:
                    st.subheader("Визуализации")
                    cols = st.columns(2)
                    for i, fname in enumerate(pngs):
                        cols[i % 2].image(os.path.join(PATHS["detective_dir"], fname),
                                          caption=fname, use_container_width=True)

            with st.expander("Полный JSON"):
                st.json(problems)
        else:
            st.info("Запусти quality-guard → audit.py")

    # Аннотации
    with tab2:
        labeled_df = load_parquet(PATHS["labeled"])
        if labeled_df is not None and "confidence" in labeled_df.columns:
            import matplotlib.pyplot as plt

            c1, c2, c3 = st.columns(3)
            c1.metric("Всего",           f"{len(labeled_df):,}")
            c2.metric("Ср. уверенность", f"{labeled_df['confidence'].mean():.2f}")
            low = (labeled_df["confidence"] < 0.7).sum()
            c3.metric("Низкая уверенность (<0.7)", f"{low:,} ({low/len(labeled_df)*100:.1f}%)")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor("#0e1117")
            for ax in (ax1, ax2):
                ax.set_facecolor("#1a1a2e")
                ax.tick_params(colors="white")
                for spine in ax.spines.values(): spine.set_edgecolor("#444")

            ax1.hist(labeled_df["confidence"], bins=20, color="#00FF88", edgecolor="#0e1117")
            ax1.set_title("Распределение уверенности", color="white")
            ax1.set_xlabel("Confidence", color="white")
            ax1.axvline(0.7, color="#FF8800", linestyle="--", label="порог 0.7")
            ax1.legend(facecolor="#1a1a2e", labelcolor="white")

            if "label" in labeled_df.columns:
                vc = labeled_df["label"].value_counts()
                ax2.barh(vc.index.tolist(), vc.values, color="#4FC3F7")
                ax2.set_title("Распределение классов", color="white")
                ax2.tick_params(colors="white")

            st.pyplot(fig)
        else:
            st.info("Запусти auto-tagger")

        review_df = load_review()
        if review_df is not None:
            st.subheader("HITL итоги")
            total = len(review_df)
            corrected = int((review_df.get("corrected_label", pd.Series()) != review_df.get("label", pd.Series())).sum())
            c1, c2 = st.columns(2)
            c1.metric("В очереди", total)
            c2.metric("Исправлено человеком", corrected)

    # Active Learning
    with tab3:
        import matplotlib.pyplot as plt

        histories = {n: load_json(p) for n, p in [
            ("entropy", PATHS["al_entropy"]),
            ("margin",  PATHS["al_margin"]),
            ("random",  PATHS["al_random"]),
        ] if exists(p)}

        if not histories:
            st.info("Запусти smart-sampler → al_cycle.py")
        else:
            metric = st.radio("Метрика", ["accuracy", "f1"], horizontal=True, key="al_metric")
            colors = {"entropy": "#00FF88", "margin": "#4FC3F7", "random": "#FF8800"}

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values(): spine.set_edgecolor("#444")

            finals = {}
            for name, hist in histories.items():
                xs = [h["n_labeled"] for h in hist]
                ys = [h.get(metric, h.get("accuracy", 0)) for h in hist]
                ax.plot(xs, ys, label=name, color=colors[name], linewidth=2, marker="o", markersize=4)
                finals[name] = ys[-1] if ys else 0

            ax.set_xlabel("Размечено примеров")
            ax.set_ylabel(metric.upper())
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            ax.grid(True, alpha=0.2, color="#444")
            st.pyplot(fig)

            c = st.columns(len(finals))
            for i, (name, val) in enumerate(finals.items()):
                c[i].metric(name, f"{val:.3f}")

            best = max(finals, key=finals.get)
            st.success(f"Лучшая стратегия: **{best}** ({metric} = {finals[best]:.3f})")

            # Сколько примеров сэкономлено
            if "entropy" in histories and "random" in histories:
                target = finals.get("entropy", 0) * 0.95
                en = next((h["n_labeled"] for h in histories["entropy"] if h.get(metric, 0) >= target), None)
                rn = next((h["n_labeled"] for h in histories["random"]  if h.get(metric, 0) >= target), None)
                if en and rn:
                    st.info(f"Для достижения {target:.2f} {metric}: entropy = **{en}** примеров, random = **{rn}**. Экономия: **{rn - en}** примеров.")

    # Модель
    with tab4:
        if exists(PATHS["model"]):
            st.success("✅ Модель обучена: `models/final_model.pkl`")
            h = load_json(PATHS["al_entropy"]) or load_json(PATHS["al_margin"])
            if h:
                st.subheader("Итоговые метрики (последняя итерация)")
                last = h[-1]
                cols = st.columns(len(last))
                for i, (k, v) in enumerate(last.items()):
                    cols[i].metric(k, f"{v:.3f}" if isinstance(v, float) else v)
        else:
            st.info("Модель ещё не обучена")

# ══════════════════════════════════════════════════════════════════════════════
# ВЫВОДЫ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Выводы":
    st.title("💡 Выводы и рекомендации")

    raw_df     = load_parquet(PATHS["raw"])
    cleaned_df = load_parquet(PATHS["cleaned"])
    labeled_df = load_parquet(PATHS["labeled"])
    problems   = load_json(PATHS["problems"])
    review_df  = load_review()

    al_histories = {n: load_json(p) for n, p in [
        ("entropy", PATHS["al_entropy"]),
        ("margin",  PATHS["al_margin"]),
        ("random",  PATHS["al_random"]),
    ] if exists(p)}

    # 1. Описание датасета
    st.subheader("1. Датасет")
    if labeled_df is not None:
        tcol_name = text_col(labeled_df)
        modalities = labeled_df["modality"].unique().tolist() if "modality" in labeled_df.columns else ["—"]
        classes    = labeled_df["label"].unique().tolist()    if "label"    in labeled_df.columns else ["—"]
        st.markdown(f"""
- **Объём (финальный):** {len(labeled_df):,} примеров
- **Модальности:** {', '.join(map(str, modalities))}
- **Классы:** {', '.join(map(str, classes))}
- **Источники:** {', '.join(labeled_df['source'].str.split(':').str[0].unique().tolist()) if 'source' in labeled_df.columns else '—'}
        """)

    # 2. Качество
    st.subheader("2. Качество данных")
    if problems and raw_df is not None and cleaned_df is not None:
        dropped = len(raw_df) - len(cleaned_df)
        st.markdown(f"""
- Найдено **{problems.get('duplicate_count', 0)}** дубликатов, **{problems.get('outlier_count', 0)}** выбросов
- Удалено **{dropped:,}** строк ({dropped/len(raw_df)*100:.1f}%)
- Дисбаланс классов: **{problems.get('imbalance_ratio', 0):.1f}×** (max/min)
        """)
        if problems.get("imbalance_ratio", 1) > 3:
            st.warning("⚠️ Высокий дисбаланс классов. Рекомендуется oversampling или взвешенная функция потерь.")

    # 3. HITL
    st.subheader("3. Human-in-the-Loop")
    if review_df is not None:
        label_col = "label" if "label" in review_df.columns else "predicted_label"
        total = len(review_df)
        corrected = int((review_df.get("corrected_label", pd.Series()) != review_df.get(label_col, pd.Series())).sum())
        error_rate = corrected / total * 100 if total > 0 else 0
        st.markdown(f"""
- На ревью отправлено: **{total}** примеров (confidence < порог)
- Исправлено человеком: **{corrected}** ({error_rate:.1f}%)
- Авто-разметка точна примерно в **{100-error_rate:.1f}%** случаев из низкоуверенных
        """)
        if error_rate > 30:
            st.warning("⚠️ Высокий % исправлений — стоит снизить порог confidence или дообучить авто-теггер.")

    # 4. Active Learning
    st.subheader("4. Active Learning")
    if al_histories:
        finals = {}
        for name, hist in al_histories.items():
            last = hist[-1]
            finals[name] = last.get("accuracy", last.get("f1", 0))
        best = max(finals, key=finals.get)
        st.markdown(f"""
- Лучшая стратегия: **{best}** (accuracy = {finals[best]:.3f})
- Результаты: {', '.join(f'{n}: {v:.3f}' for n, v in finals.items())}
        """)
        if "entropy" in finals and "random" in finals:
            diff = finals["entropy"] - finals["random"]
            if diff > 0.01:
                st.success(f"Entropy sampling даёт +{diff*100:.1f}% к точности по сравнению со случайным выбором.")

    # 5. Ретроспектива
    st.subheader("5. Ретроспектива")
    st.markdown("""
**Что сработало:**
- Унифицированная схема (text/label/modality/source) позволила легко переключаться между источниками
- Авто-разметка через LLM значительно ускорила подготовку данных
- Active Learning сократил количество примеров, необходимых для достижения целевой точности

**Что можно улучшить:**
- Добавить более сложные модели (fine-tuned transformers) вместо LogReg/SVM
- Расширить поддержку изображений (ResNet embeddings вместо гистограмм)
- Автоматизировать запуск HITL-проверки при появлении новых данных
    """)
