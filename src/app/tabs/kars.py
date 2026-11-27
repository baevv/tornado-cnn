"""Model Karşılaştırma: ROC, PR, performans ve güvenilirlik diyagramları, EF ve uzaklığa göre başarım,
güven aralıklı ölçüt tablosu + NC / RC / WC, karmaşıklık matrisleri, öğrenme eğrileri, hiperparametre tablosu.

Seed ensemble'ları (O3) yalnız bu sekmede, ayrı birer model olarak seçilebilir.
"""
import numpy as np
import pandas as pd
import streamlit as st

from src.app import cache
from src.app import evaluate as E
from src.app import plots as P
from src.helper import evaluate as ev

CFG = {"displayModeBar": False}
ENS_COLORS = ["#0B3C5D", "#7A1F5C", "#3D5A1E", "#5C3B0B"]
# Varsayılan seçim: ana karşılaştırma (ensemble'lar) ve öne çıkan ablasyonlar; hiçbiri yoksa bütün modeller
DEFAULT_PICK = ["03_baseline_ens", "05_resnet18_pretrained_ens", "05_cnn_coord", "05_cnn_bn", "03_mit"]


def all_models(models):
    return E.with_ensembles(models, cache.store()[0])


def _colored(models):
    ens = [m for m in models if m.members]
    for i, m in enumerate(ens):
        m.color = ENS_COLORS[i % len(ENS_COLORS)]
    return models


def defaults(ss, models):
    keys = [m.key for m in all_models(models)]
    pick = [k for k in DEFAULT_PICK if k in keys]
    ss.setdefault("ks_models", pick if len(pick) >= 2 else [m.key for m in models])
    ss.setdefault("ks_chart", "roc")
    ss.setdefault("ks_split", "val")


def handle(ev_, ss):
    if ev_["a"] == "kars_models":
        ss.ks_models = list(ev_["keys"])
    elif ev_["a"] == "kars_chart":
        ss.ks_chart = ev_["chart"]
    elif ev_["a"] == "kars_split":
        ss.ks_split = ev_["split"]


def state(ss, store, models) -> dict:
    items = [{"id": m.key, "dot": m.color, "name": m.label,
              "tip": m.long_label + (f" · eşik {m.threshold:.3f}" if m.members else "")}
             for m in _colored(all_models(models))]
    return {"models": ss.ks_models, "chart": ss.ks_chart, "split": ss.ks_split, "items": items}


def _scored(chosen, store, split):
    """[(model, örnekler, skorlar)] ve skoru olmayanların adları."""
    out, missing = [], []
    for m in chosen:
        s, p = E.split_scores(m, store, split)
        if s is None or s["label"].nunique() < 2:
            missing.append(m.label)
        else:
            out.append((m, s, p))
    return out, missing


def body(ss, store, models):
    chosen = [m for m in _colored(all_models(models)) if m.key in ss.ks_models]
    if not chosen:
        st.html("<div class='note'>Karşılaştırmak için üstten en az bir model seç.</div>")
        return
    ch, split = ss.ks_chart, ss.ks_split
    if ch in ("roc", "pr", "perf", "rel", "ef", "range", "table"):
        scored, missing = _scored(chosen, store, split)
        if scored:
            CHARTS[ch](scored, store, split)
        if missing:
            st.html(E.no_scores_note(split).replace("Bu bölüm", ", ".join(missing) + " · bu bölüm"))
    elif ch == "cm":
        cols = st.columns(min(len(chosen), 3))
        for i, m in enumerate(chosen):
            met = E.metrics_for(m, store, split)
            with cols[i % len(cols)]:
                if met and "TP" in met:
                    st.plotly_chart(P.confusion(met["TP"], met["FP"], met["FN"], met["TN"], title=m.label, color=m.color, height=300),
                                    use_container_width=True, config=CFG)
                else:
                    st.html(f"<div class='h-panel'>{m.label}</div>" + E.no_scores_note(split))
    elif ch == "learn":
        learn(chosen)
    elif ch == "hp":
        st.dataframe(hp_table([m for m in chosen if not m.members], store, split), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- grafikler

def roc(scored, store, split):
    from sklearn.metrics import roc_auc_score, roc_curve
    curves = []
    for m, s, p in scored:
        y = s["label"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        curves.append((m.label, m.color, fpr, tpr, roc_auc_score(y, p)))
    st.plotly_chart(P.roc(curves), use_container_width=True, config=CFG)


def pr(scored, store, split):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    curves, base = [], None
    for m, s, p in scored:
        y = s["label"].to_numpy()
        base = float(y.mean())
        prec, rc, _ = precision_recall_curve(y, p)
        curves.append((m.label, m.color, rc, prec, average_precision_score(y, p)))
    st.plotly_chart(P.pr(curves, base), use_container_width=True, config=CFG)


def perf(scored, store, split):
    st.plotly_chart(P.perf_diagram([(m.label, m.color, s["label"].to_numpy(), p, m.threshold) for m, s, p in scored]),
                    use_container_width=True, config=CFG)
    st.html("<div class='note'>Sağ üst köşe iyidir. Dolgu CSI, kesikli doğrular frekans yanlılığıdır (Bias; 1 = "
            "gerçekte olduğu kadar hortum der). Nokta, doğrulamada seçilen eşikteki karardır.</div>")


def rel(scored, store, split):
    st.plotly_chart(P.reliability([(m.label, m.color, s["label"].to_numpy(), p) for m, s, p in scored]),
                    use_container_width=True, config=CFG)
    st.html("<div class='note'>Köşegen tam kalibrasyondur. <code>pos_weight</code> hortum sınıfını yaklaşık 12 kat "
            "ağırlıklandırdığı için eğriler köşegenin altında kalır: skor, gerçek hortum oranından yüksektir. "
            "Bu yüzden arayüz bu değeri olasılık değil <b>skor</b> olarak gösterir.</div>")


def ef_chart(scored, store, split):
    groups, tables = [], {}
    for m, s, p in scored:
        mt = E.meta(s)
        groups.append((m.label, m.color, ev.ef_class(mt), p, m.threshold))
        tables[m.label] = ev.ef_table(mt, p, m.threshold)
    st.plotly_chart(P.ef_box(groups), use_container_width=True, config=CFG)
    t = pd.concat({k: v["eşiği geçen oran"] for k, v in tables.items()}, axis=1).round(3)
    t.insert(0, "n", next(iter(tables.values()))["n"])
    st.html("<div class='note'>Tabloda eşiği geçen oran: TOR sınıflarında yakalama (POD), WRN ve NUL'da yanlış "
            "alarm oranı.</div>")
    st.dataframe(t, use_container_width=True)


def range_chart(scored, store, split):
    tables = [(m.label, m.color, ev.range_table(E.meta(s), p, m.threshold)) for m, s, p in scored]
    st.plotly_chart(P.range_lines(tables), use_container_width=True, config=CFG)
    st.html("<div class='note'>Uzaklık, pencere merkezinin radara uzaklığıdır (rng_min + 30 km). Aralıklar arası "
            "kıyasta hortum oranından bağımsız olan ROC-AUC esas alınır. Uzak aralıkta hortum sayısı azdır; o "
            "aralığa temkinli bakın.</div>")


def table(scored, store, split):
    t = E.boot_table([m for m, _, _ in scored], store, split)
    if not t.empty:
        a = t.attrs
        head = (f"{a['n']:,} kare · {a['storms']:,} fırtına · hortum oranı {a['base_rate']:.3f} (rastgele modelin "
                f"PR-AUC'si) · %95 GA fırtına bazlı bootstrap ({E.N_BOOT} örneklem)").replace(",", ".")
        if a.get("n_ref"):
            head += (f"<br>Δ referansı: temel modelin {a['n_ref']} seed ortalaması {a['ref']:.3f}, seed std "
                     f"σ = {a['sd']:.3f}. Yorum: <b>fark</b> = GA sıfırı dışlar ve |Δ| &gt; 2σ · <b>eğilim</b> = "
                     "|Δ| &gt; σ · <b>gürültü</b> = diğer")
        st.html(f"<div class='note'>{head}</div>")
        shown = [m.label for m, _, _ in scored]
        st.dataframe(t[t["model"].isin(shown)], use_container_width=True, hide_index=True)
    v = E.views_table([m for m, _, _ in scored], store, split)
    if not v.empty:
        rates = " · ".join(f"{k} {r:.3f}" for k, r in v.attrs["rates"].items())
        st.html("<div class='h-panel'>NC / RC / WC görünümleri</div><div class='note'>NC: TOR ↔ WRN + NUL (genel) · "
                "RC: TOR ↔ NUL (kolay negatifler) · WC: TOR ↔ WRN (zor negatifler). Hortum oranı: "
                f"{rates}. PR-AUC görünümler arasında değil, aynı görünümde modeller arasında kıyaslanır.</div>")
        st.dataframe(v, use_container_width=True, hide_index=True)


CHARTS = {"roc": roc, "pr": pr, "perf": perf, "rel": rel, "ef": ef_chart, "range": range_chart, "table": table}


def learn(chosen):
    series_pr, series_loss = [], []
    for m in chosen:
        h = E.history_df(m)
        if h.empty:
            continue
        col = "doğrulama PR-AUC" if "doğrulama PR-AUC" in h else "doğrulama AUC"
        series_pr.append((m.label, m.color, h["tur"], h[col], False))
        series_loss.append((m.label, m.color, h["tur"], h["doğrulama kaybı"], False))
    if not series_pr:
        st.html("<div class='note'>Seçili modellerin öğrenme geçmişi yok (ensemble'ların geçmişi olmaz).</div>")
        return
    c1, c2 = st.columns(2)
    c1.plotly_chart(P.learning(series_pr, "PR-AUC", "Doğrulama PR-AUC (izlenen ölçüt)"), use_container_width=True, config=CFG)
    c2.plotly_chart(P.learning(series_loss, "kayıp", "Doğrulama kaybı"), use_container_width=True, config=CFG)


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
