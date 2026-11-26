"""Model Karşılaştırma: ROC, PR, karmaşıklık matrisleri, öğrenme eğrileri, hiperparametre tablosu."""
import numpy as np
import pandas as pd
import streamlit as st

from src.app import evaluate as E
from src.app import plots as P

CFG = {"displayModeBar": False}


def defaults(ss, models):
    ss.setdefault("ks_models", [m.key for m in models])
    ss.setdefault("ks_chart", "roc")
    ss.setdefault("ks_split", "val")


def handle(ev, ss):
    if ev["a"] == "kars_models":
        ss.ks_models = list(ev["keys"])
    elif ev["a"] == "kars_chart":
        ss.ks_chart = ev["chart"]
    elif ev["a"] == "kars_split":
        ss.ks_split = ev["split"]


def state(ss, store, models) -> dict:
    return {"models": ss.ks_models, "chart": ss.ks_chart, "split": ss.ks_split}


def body(ss, store, models):
    chosen = [m for m in models if m.key in ss.ks_models]
    if not chosen:
        st.html("<div class='note'>Karşılaştırmak için üstten en az bir model seç.</div>")
        return
    ch = ss.ks_chart
    if ch in ("roc", "pr"):
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
        curves, base, missing = [], None, []
        for m in chosen:
            s, p = E.split_scores(m, store, ss.ks_split)
            if s is None or s["label"].nunique() < 2:
                missing.append(m.label)
                continue
            y = s["label"].to_numpy()
            base = float(y.mean())
            if ch == "roc":
                fpr, tpr, _ = roc_curve(y, p)
                curves.append((m.label, m.color, fpr, tpr, roc_auc_score(y, p)))
            else:
                pr, rc, _ = precision_recall_curve(y, p)
                curves.append((m.label, m.color, rc, pr, average_precision_score(y, p)))
        if curves:
            st.plotly_chart(P.roc(curves) if ch == "roc" else P.pr(curves, base), use_container_width=True, config=CFG)
        if missing:
            st.html(E.no_scores_note(ss.ks_split).replace("Bu bölüm", ", ".join(missing) + " · bu bölüm"))
    elif ch == "cm":
        cols = st.columns(min(len(chosen), 3))
        for i, m in enumerate(chosen):
            met = E.metrics_for(m, store, ss.ks_split)
            with cols[i % len(cols)]:
                if met and "TP" in met:
                    st.plotly_chart(P.confusion(met["TP"], met["FP"], met["FN"], met["TN"], title=m.label, color=m.color, height=300),
                                    use_container_width=True, config=CFG)
                else:
                    st.html(f"<div class='h-panel'>{m.label}</div>" + E.no_scores_note(ss.ks_split))
    elif ch == "learn":
        series_auc, series_loss = [], []
        for m in chosen:
            h = E.history_df(m)
            if h.empty:
                continue
            series_auc.append((m.label, m.color, h["tur"], h["doğrulama AUC"], False))
            series_loss.append((m.label, m.color, h["tur"], h["doğrulama kaybı"], False))
        c1, c2 = st.columns(2)
        c1.plotly_chart(P.learning(series_auc, "ROC-AUC", "Doğrulama AUC"), use_container_width=True, config=CFG)
        c2.plotly_chart(P.learning(series_loss, "kayıp", "Doğrulama kaybı"), use_container_width=True, config=CFG)
    elif ch == "hp":
        st.dataframe(hp_table(chosen, store, ss.ks_split), use_container_width=True, hide_index=True)


def hp_table(models, store, split) -> pd.DataFrame:
    rows = []
    for m in models:
        c = m.results.get("config", {})
        met = E.metrics_for(m, store, split) or {}
        rows.append({
            "model": m.label, "parametre": m.params, "optimizer": c.get("optimizer"), "lr": c.get("lr"),
            "batch": c.get("batch_size"), "dropout": c.get("dropout"), "L2": c.get("weight_decay"),
            "ağırlık": c.get("weighting"), "en iyi tur": m.results.get("best_epoch"),
            "eşik": round(m.threshold, 3),
            **{k: (round(met[k], 3) if isinstance(met.get(k), (float, np.floating)) else met.get(k))
               for k in ("ROC-AUC", "PR-AUC", "F1", "recall (yakalama)", "precision (isabet)")},
            "yıllar": ", ".join(map(str, m.results.get("years", []))),
        })
    return pd.DataFrame(rows)
