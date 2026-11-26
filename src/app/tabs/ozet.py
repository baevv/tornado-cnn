"""Özet: seçili modelin ölçütleri (araç çubuğunda), karmaşıklık matrisi, öğrenme eğrisi, skor dağılımı."""
import streamlit as st

from src.app import evaluate as E
from src.app import plots as P


def defaults(ss, models):
    ss.setdefault("oz_model", ss.sel_model)
    ss.setdefault("oz_split", "val")


def handle(ev, ss):
    if ev["a"] == "ozet_model":
        ss.oz_model = ev["key"]
    elif ev["a"] == "ozet_split":
        ss.oz_split = ev["split"]


def state(ss, store, models) -> dict:
    info = next((m for m in models if m.key == ss.oz_model), None)
    return {"model": ss.oz_model, "split": ss.oz_split, "kpis": E.kpis(E.metrics_for(info, store, ss.oz_split)) if info else []}


def body(ss, store, models):
    info = next((m for m in models if m.key == ss.oz_model), None)
    if info is None:
        st.html("<div class='note'>Model bulunamadı. Model klasörleri <code>models/&lt;ad&gt;/</code> altında "
                "<code>best.pt</code> + <code>results.json</code> içermeli.</div>")
        return
    m = E.metrics_for(info, store, ss.oz_split)
    c1, c2 = st.columns([2, 3])
    with c1:
        if m and "TP" in m:
            st.plotly_chart(P.confusion(m["TP"], m["FP"], m["FN"], m["TN"], color=info.color, height=400),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.html(E.no_scores_note(ss.oz_split))
    with c2:
        h = E.history_df(info)
        if not h.empty:
            series = [("eğitim kaybı", "#8791A1", h["tur"], h["eğitim kaybı"], True),
                      ("doğrulama kaybı", info.color, h["tur"], h["doğrulama kaybı"], False)]
            st.plotly_chart(P.learning(series, "kayıp", "Öğrenme eğrisi"), use_container_width=True, config={"displayModeBar": False})
    s, p = E.split_scores(info, store, ss.oz_split)
    if s is not None:
        st.plotly_chart(P.score_hist(p, s["category"].to_numpy(), info.threshold, "Skor dağılımı · kategoriye göre"),
                        use_container_width=True, config={"displayModeBar": False})
