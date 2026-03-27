"""
ML Pipeline Dashboard — Streamlit
Запуск: streamlit run dashboard.py
"""
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="🔬",
    layout="wide",
)

# ── Пути ──────────────────────────────────────────────────────────────────────
RAW       = "data/raw/combined.parquet"
CLEANED   = "data/cleaned/cleaned.parquet"
LABELED   = "data/labeled/labeled_final.parquet"
PROBLEMS  = "data/detective/problems.json"
EDA_STATS = "data/eda/stats.json"
AL_FILES  = {
    "entropy": "data/active/history_entropy.json",
    "margin":  "data/active/history_margin.json",
    "random":  "data/active/history_random.json",
}
REVIEW_Q  = "review_queue.csv"
MODEL     = "models/final_model.pkl"

# ── Хелперы ───────────────────────────────────────────────────────────────────
def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

def load_df(path):
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

def exists_icon(path):
    return "✅" if os.path.exists(path) else "⬜"

# ── Сайдбар ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 ML Pipeline")
page = st.sidebar.radio(
    "Раздел",
    ["Обзор", "HITL Разметка", "Качество данных", "Active Learning"],
)

# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА: ОБЗОР
# ══════════════════════════════════════════════════════════════════════════════
if page == "Обзор":
    st.title("Статус пайплайна")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(f"{exists_icon(RAW)} Сбор", "combined.parquet" if os.path.exists(RAW) else "—")
    col2.metric(f"{exists_icon(CLEANED)} Чистка", "cleaned.parquet" if os.path.exists(CLEANED) else "—")
    col3.metric(f"{exists_icon(LABELED)} Разметка", "labeled_final.parquet" if os.path.exists(LABELED) else "—")
    col4.metric(f"{exists_icon(AL_FILES['entropy'])} Active Learning", "history_*.json" if os.path.exists(AL_FILES["entropy"]) else "—")
    col5.metric(f"{exists_icon(MODEL)} Модель", "final_model.pkl" if os.path.exists(MODEL) else "—")

    st.divider()

    # Ключевые числа из артефактов
    cols = st.columns(3)

    raw_df = load_df(RAW)
    if raw_df is not None:
        with cols[0]:
            st.subheader("📦 Сырые данные")
            st.metric("Примеров", f"{len(raw_df):,}")
            if "modality" in raw_df.columns:
                st.dataframe(raw_df["modality"].value_counts().rename("кол-во"), use_container_width=True)
            if "source" in raw_df.columns:
                st.caption("Источники: " + ", ".join(raw_df["source"].str.split(":").str[0].unique()))

    cleaned_df = load_df(CLEANED)
    if cleaned_df is not None:
        with cols[1]:
            st.subheader("🧹 После чистки")
            st.metric("Примеров", f"{len(cleaned_df):,}")
            if raw_df is not None:
                dropped = len(raw_df) - len(cleaned_df)
                st.metric("Удалено", f"{dropped:,}", delta=f"-{dropped/len(raw_df)*100:.1f}%", delta_color="inverse")

    labeled_df = load_df(LABELED)
    if labeled_df is not None:
        with cols[2]:
            st.subheader("🏷️ Размеченные")
            st.metric("Примеров", f"{len(labeled_df):,}")
            if "label" in labeled_df.columns:
                st.dataframe(labeled_df["label"].value_counts().rename("кол-во"), use_container_width=True)
            if "confidence" in labeled_df.columns:
                avg_conf = labeled_df["confidence"].mean()
                st.metric("Ср. уверенность", f"{avg_conf:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА: HITL РАЗМЕТКА
# ══════════════════════════════════════════════════════════════════════════════
elif page == "HITL Разметка":
    st.title("✏️ Human-in-the-Loop: проверка меток")

    if not os.path.exists(REVIEW_Q):
        st.info("Файл review_queue.csv не найден. Запусти auto-tagger чтобы создать очередь.")
        st.stop()

    df = pd.read_csv(REVIEW_Q)
    if "corrected_label" not in df.columns:
        df["corrected_label"] = df.get("label", "")

    # Классы
    all_labels = df["label"].dropna().unique().tolist()
    classes = sorted(set(l for l in all_labels if str(l) != "unlabeled"))

    # Фильтр
    show_only_pending = st.checkbox("Показывать только непроверенные", value=True)
    if show_only_pending:
        view = df[df["corrected_label"] == df["label"]].copy()
    else:
        view = df.copy()

    total = len(df)
    pending = int((df["corrected_label"] == df["label"]).sum())
    done = total - pending

    p1, p2, p3 = st.columns(3)
    p1.metric("Всего", total)
    p2.metric("Проверено", done)
    p3.metric("Осталось", pending)
    st.progress(done / total if total > 0 else 0)

    if view.empty:
        st.success("Все примеры проверены!")
        st.stop()

    st.divider()

    # Постраничный навигатор
    if "hitl_idx" not in st.session_state:
        st.session_state.hitl_idx = 0

    idx = st.session_state.hitl_idx
    idx = min(idx, len(view) - 1)
    row = view.iloc[idx]
    real_idx = view.index[idx]

    modality = str(row.get("modality", "text")) if "modality" in row.index else "text"
    text = str(row.get("text", ""))
    current_label = str(row.get("label", ""))
    confidence = float(row.get("confidence", 0.0))

    col_nav, col_counter = st.columns([3, 1])
    with col_counter:
        st.caption(f"Пример {idx + 1} / {len(view)}")

    # Контент
    st.markdown(f"**Модальность:** `{modality}` &nbsp;&nbsp; **Уверенность:** `{confidence:.2f}`")

    if modality == "image":
        if os.path.exists(text):
            st.image(text, width=400)
        else:
            st.code(text)
    elif modality == "audio":
        if os.path.exists(text):
            st.audio(text)
        else:
            st.code(text)
    else:
        st.text_area("Текст", value=text[:2000], height=150, disabled=True)

    # Выбор метки
    current_corrected = str(row.get("corrected_label", current_label))
    default_idx = classes.index(current_corrected) if current_corrected in classes else (
        classes.index(current_label) if current_label in classes else 0
    )

    chosen = st.radio(
        "Правильная метка:",
        classes,
        index=default_idx,
        horizontal=True,
        key=f"label_{real_idx}",
    )

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("✅ Сохранить", type="primary"):
            df.at[real_idx, "corrected_label"] = chosen
            df.to_csv(REVIEW_Q, index=False, encoding="utf-8-sig")
            st.session_state.hitl_idx = min(idx + 1, len(view) - 1)
            st.rerun()
    with c2:
        if st.button("⏭️ Пропустить"):
            st.session_state.hitl_idx = min(idx + 1, len(view) - 1)
            st.rerun()

    # Навигация
    st.divider()
    n1, n2 = st.columns(2)
    with n1:
        if st.button("← Назад") and idx > 0:
            st.session_state.hitl_idx -= 1
            st.rerun()
    with n2:
        if st.button("Вперёд →") and idx < len(view) - 1:
            st.session_state.hitl_idx += 1
            st.rerun()

    # Таблица всех записей
    with st.expander("Все записи в очереди"):
        st.dataframe(df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА: КАЧЕСТВО ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Качество данных":
    st.title("🔍 Отчёт качества данных")

    problems = load_json(PROBLEMS)
    if problems is None:
        st.info("Запусти quality-guard → audit.py чтобы получить отчёт.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Пропуски", problems.get("missing_count", 0))
    c2.metric("Дубликаты", problems.get("duplicate_count", 0))
    c3.metric("Выбросы", problems.get("outlier_count", 0))
    imbalance = problems.get("imbalance_ratio", None)
    c4.metric("Дисбаланс (max/min)", f"{imbalance:.1f}x" if imbalance else "—")

    # PNG-визуализации из data/detective/
    detective_dir = "data/detective"
    if os.path.exists(detective_dir):
        pngs = [f for f in os.listdir(detective_dir) if f.endswith(".png")]
        if pngs:
            st.subheader("Визуализации")
            cols = st.columns(min(len(pngs), 2))
            for i, fname in enumerate(sorted(pngs)):
                cols[i % 2].image(os.path.join(detective_dir, fname), caption=fname, use_container_width=True)

    with st.expander("Полный JSON отчёта"):
        st.json(problems)

# ══════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА: ACTIVE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Active Learning":
    st.title("📈 Active Learning — кривые обучения")

    found = {name: load_json(path) for name, path in AL_FILES.items() if os.path.exists(path)}

    if not found:
        st.info("Запусти smart-sampler → al_cycle.py чтобы получить историю.")
        st.stop()

    import matplotlib.pyplot as plt

    metric = st.radio("Метрика", ["accuracy", "f1"], horizontal=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    colors = {"entropy": "#00FF88", "margin": "#4FC3F7", "random": "#FF8800"}
    final_metrics = {}

    for name, history in found.items():
        xs = [h["n_labeled"] for h in history]
        ys = [h.get(metric, h.get("accuracy", 0)) for h in history]
        ax.plot(xs, ys, label=name, color=colors.get(name, "white"), linewidth=2, marker="o", markersize=4)
        final_metrics[name] = ys[-1] if ys else 0

    ax.set_xlabel("Размечено примеров")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Learning curve — {metric.upper()}")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#444")
    st.pyplot(fig)

    st.subheader("Итоговые метрики")
    cols = st.columns(len(final_metrics))
    for i, (name, val) in enumerate(final_metrics.items()):
        cols[i].metric(name, f"{val:.3f}")

    if found:
        best = max(final_metrics, key=final_metrics.get)
        st.success(f"Лучшая стратегия: **{best}** ({metric} = {final_metrics[best]:.3f})")

    # Сколько примеров сэкономлено
    if "entropy" in final_metrics and "random" in final_metrics:
        entropy_hist = found["entropy"]
        random_hist  = found["random"]
        target = final_metrics.get("entropy", 0) * 0.95
        entropy_n = next((h["n_labeled"] for h in entropy_hist if h.get(metric, 0) >= target), None)
        random_n  = next((h["n_labeled"] for h in random_hist  if h.get(metric, 0) >= target), None)
        if entropy_n and random_n:
            saved = random_n - entropy_n
            st.info(f"Entropy достигла {target:.2f} {metric} при **{entropy_n}** примерах, "
                    f"random — при **{random_n}**. Сэкономлено: **{saved}** примеров.")
