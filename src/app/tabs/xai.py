"""Açıklanabilirlik: modelin kararları neye dayanıyor?

İki görünüm:
  Tek kare  Seçili karede Grad-CAM / Integrated Gradients / Occlusion haritası, en güçlü dönme ile açıklama zirvesi,
            kanal SHAP şelalesi, silme eğrisi (sadakat testi) ve bu ölçülerden üretilen yorum.
            Kare Gezgin'deki "Bu kareyi açıkla" ya da Hata Analizi'ndeki "Açıkla" düğmesiyle seçilir.
  Genel     models/<ad>/xai.parquet'ten (scripts/make_xai.py) toplu sonuçlar: karar türüne göre kanal katkısı,
            zirve ↔ dönme mesafesi, model × yöntem sadakat tablosu, sağlamlık kontrolü, tipik örnekler.

Yorum cümleleri sabit metin değildir; her biri ekrandaki ölçülerden kurallarla üretilir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.app import cache
from src.app import data as D
from src.app import linked
from src.app import plots as P
from src.app import theme as T
from src.helper.xai import METHODS

CFG = {"displayModeBar": False}
XAI_FILE = "xai.parquet"
KIND_TR = {"tp": "Doğru pozitif", "fn": "Kaçırılan", "fp": "Yanlış alarm", "tn": "Doğru negatif"}
KIND_COLORS = {"tp": T.GOOD, "fn": T.TOR, "fp": T.ACCENT, "tn": T.NUL}
CAT_TR = {"TOR": "Hortum", "WRN": "Uyarı", "NUL": "Fırtına"}
NEAR_KM = 3.0                      # "dönmenin üstünde" sayılan mesafe
METHOD_DESC = {
    "cam": "Grad-CAM son evrişim katmanının çıktısını skorun gradyanıyla ağırlıklandırır ve girdi boyutuna büyütür. "
           "Modelin baktığı bölgeyi gösterir ama kabadır (TornadoCNN'de 15 × 30, ResNet-18'de 4 × 8 hücre).",
    "ig": "Integrated Gradients boş girdiden (yankı yok) bu kareye 32 adımda gradyanları toplar. Piksel ve kanal "
          "düzeyindedir; harita gösterim için hafifçe yumuşatıldı.",
    "occ": "Occlusion her seferinde 10 × 10 piksellik bir yamayı boş değerle kapatır ve skorun ne kadar düştüğünü ölçer "
           "(288 ileri geçiş). Modelin içine bakmaz; doğrudan etkiyi gösterir.",
    "loc": "Konum haritası modelin kendi sınıflandırıcısını, global havuzlamadan önceki özellik haritasının her konumuna "
           "ayrı ayrı uygular (TornadoCNN'de 7 × 15, ResNet-18'de 4 × 8): \"bu konum tek başına olsaydı skor ne olurdu?\". "
           "Değerler skor ölçeğindedir. Global max havuzlama her kanalın en büyüğünü farklı konumlardan alabildiği için "
           "kararın yalnızca bir yaklaşığıdır.",
}
# 01 defteri D3: tek değişkenli ayırt edicilik (AUC, merkezde özet ölçü). TOR ↔ diğerleri ve TOR ↔ WRN.
EDA_AUC = {"DBZ": 0.624, "VEL": 0.764, "WIDTH": 0.685}
EDA_AUC_WRN = {"DBZ": 0.520, "VEL": 0.685, "WIDTH": 0.582}
SHAP_FILE, ROT_FILE = "shap_all.parquet", "rotation.parquet"


def fmt(v, d=2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


# ---------------------------------------------------------------- durum

def defaults(ss, models):
    ss.setdefault("ax_model", ss.sel_model)
    ss.setdefault("ax_view", "frame")
    ss.setdefault("ax_method", "cam")
    ss.setdefault("ax_ep", None)
    ss.setdefault("ax_frame", None)
    ss.setdefault("ax_kind", "fp")


def open_frame(ss, ep_id: int, frame: int, model_key=None):
    """Başka sekmeden (Gezgin, Hata Analizi) bir kareyi açıklamak için."""
    ss.tab, ss.ax_view, ss.ax_ep, ss.ax_frame = "xai", "frame", int(ep_id), int(frame)
    if model_key:
        ss.ax_model = model_key


def handle(ev, ss, store=None):
    a = ev["a"]
    if a == "xai_model":
        ss.ax_model = ev["key"]
    elif a == "xai_view":
        ss.ax_view = ev["view"]
    elif a == "xai_method":
        ss.ax_method = ev["method"]
    elif a == "xai_frame" and ss.ax_frame is not None:
        ss.ax_frame = int(ss.ax_frame) + int(ev["step"])        # sınırlar state() içinde
    elif a == "xai_open" and ss.ax_ep is not None:
        ss.tab, ss.gz_ep, ss.gz_filter, ss.gz_split, ss.gz_frame = "gezgin", int(ss.ax_ep), "all", "all", ss.ax_frame
        ss.sel_model = ss.ax_model


def _resolve_frame(ss, store):
    """Açıklanacak fırtına ve kare; seçilmemişse Gezgin'deki fırtına, o da yoksa testteki ilk hortumlu fırtına."""
    if store is None:
        return None
    if ss.ax_ep is None or ss.ax_ep not in store.episodes.index:
        ep = ss.get("gz_ep")
        if ep is None or ep not in store.episodes.index:
            ids = D.pool(store.episodes, "tor", "test")
            ep = ids[0] if len(ids) else store.episodes.index[0]
        ss.ax_ep, ss.ax_frame = int(ep), None
    rows = store.frames(ss.ax_ep)
    if ss.ax_frame is None:
        cats = rows["category"].tolist()
        ss.ax_frame = cats.index("TOR") if "TOR" in cats else len(cats) - 1
    ss.ax_frame = int(np.clip(ss.ax_frame, 0, len(rows) - 1))
    return rows


def state(ss, store, models) -> dict:
    if ss.ax_model not in {m.key for m in models} and models:
        ss.ax_model = models[0].key
    pos = None
    rows = _resolve_frame(ss, store) if ss.ax_view == "frame" else None
    if rows is not None:
        r = rows.iloc[ss.ax_frame]
        pos = f"#{ss.ax_ep} · kare {ss.ax_frame + 1}/{len(rows)} · {r['category']}"
    return {"model": ss.ax_model, "view": ss.ax_view, "method": ss.ax_method, "pos": pos, "split": "test"}


def body(ss, store, models):
    info = next((m for m in models if m.key == ss.ax_model), None)
    if info is None:
        st.html("<div class='note'>Model bulunamadı.</div>")
        return
    if ss.ax_view == "global":
        _global(ss, store, models, info)
    else:
        _frame(ss, store, info)


# ---------------------------------------------------------------- tek kare

def _strip(r, ep_id, score, thr) -> str:
    cat = r["category"]
    col = {"TOR": T.TOR, "WRN": T.WRN}.get(cat, T.NUL)
    y = int(r["label"])
    fire = score >= thr
    kind = ("tp" if fire else "fn") if y else ("fp" if fire else "tn")
    ef = f"EF{int(r['ef'])} · " if cat == "TOR" and r["ef"] >= 0 else ""
    cells = [("Episode", ep_id), ("Tarih · saat", f"{r['end_time']:%Y-%m-%d %H:%M}"), ("İstasyon", r.get("site", "")),
             ("Bölüm", r["split"])]
    return (f"<div class='xs' style='--st:{col}'>"
            f"<div class='pl'><b>{CAT_TR.get(cat, cat)}</b><span>{ef}{r['end_time']:%Y}</span></div>"
            + "".join(f"<div class='c'><span class='k'>{k}</span><span class='v'>{v}</span></div>" for k, v in cells)
            + "<div class='pr'>"
            f"<div class='c'><span class='k'>Model skoru</span><span class='v'>{fmt(score, 3)}</span></div>"
            f"<div class='c'><span class='k'>Eşik</span><span class='v'>{fmt(thr, 3)}</span></div>"
            f"<div class='c'><span class='k'>Karar</span><span class='v' style='color:{KIND_COLORS[kind]}'>{KIND_TR[kind]}</span></div>"
            "</div></div>")


def _edge(p) -> bool:
    return p is not None and (p[0] < 5 or p[0] > 114 or p[1] < 10 or p[1] > 229)


def frame_bullets(e: dict, variables, method: str, thr: float, rot) -> list:
    """Ölçülerden yorum cümleleri: (renk, metin)."""
    out = []
    score, phi = e["score"], np.asarray(e["phi"])
    tot = np.abs(phi).sum()
    if tot > 1e-6:
        i = int(np.argmax(np.abs(phi)))
        v, share = variables[i], abs(phi[i]) / tot
        if phi[i] >= 0 and score < thr:
            msg = (f"Skoru en çok yükselten kanal {v} ({phi[i]:+.3f}, kanal katkılarındaki payı %{100 * share:.0f}); "
                   f"yine de toplam skor ({fmt(score, 3)}) eşiğin altında kalıyor.")
        elif phi[i] >= 0:
            msg = f"Model bu karede kararını en çok {v} kanalına dayandırıyor (kanal katkılarındaki payı %{100 * share:.0f}, {phi[i]:+.3f})."
            if v == "VEL":
                msg += " Hortumun radardaki imzası radyal hızdaki dönme çifti olduğu için bu beklenen davranış."
            elif v == "WIDTH":
                msg += " Spektrum genişliği türbülansı gösterir; dönmeden bağımsız da yükselebilir."
        else:
            msg = f"{v} kanalı skoru en çok düşüren kanal ({phi[i]:+.3f}); bu kanal olmasa skor daha yüksek olurdu."
        neg = [variables[k] for k in range(len(phi)) if phi[k] < -0.02 and k != i]
        if neg:
            msg += f" {', '.join(neg)} ise skoru düşürüyor."
        out.append((P.CHANNEL_COLORS.get(v, T.MUTED), msg))
    else:
        out.append((T.NUL, "Kanalların hiçbiri skoru belirgin değiştirmiyor; model boş girdiye verdiği skoru veriyor."))

    name, p, d = P.METHOD_NAMES[method], e["peaks"][method], e["dist_km"][method]
    if rot is None:
        out.append((T.INK, f"Bu karenin merkezinde belirgin bir dönme bulunamadı; {name} zirvesi dönmeyle karşılaştırılamıyor."))
    elif p is None:
        out.append((T.INK, f"{name} haritası bu karede boş (skoru artıran bölge yok)."))
    else:
        if d <= NEAR_KM:
            msg = f"{name} zirvesi en güçlü dönmeye {fmt(d, 1)} km uzaklıkta: model dönme imzasının üstüne bakıyor."
        elif d <= 10:
            msg = f"{name} zirvesi en güçlü dönmeye {fmt(d, 1)} km uzaklıkta: yakın ama tam üstünde değil."
        else:
            msg = f"{name} zirvesi en güçlü dönmeden {fmt(d, 1)} km uzakta: model dönme dışında bir yapıya bakıyor."
        if _edge(p):
            msg += " Zirve görüntünün kenarında; bu, verideki bir kenar etkisine dayanıldığını düşündürür."
        out.append((T.INK, msg))

    if method == "loc" and e.get("loc_gap") is not None:
        g = e["loc_gap"]
        out.append((P.METHOD_COLORS["loc"], f"Konum haritası bir yaklaşıktır: gerçek logit ile haritanın en büyüğü "
                                            f"arasındaki fark {fmt(g, 2)} ({'küçük, harita kararı iyi temsil ediyor' if abs(g) < 0.5 else 'büyük, harita kararı yalnızca kabaca temsil ediyor'})."))
    a, ar = e["auc"][method], e["auc"]["rnd"]
    if score < thr:
        out.append((P.METHOD_COLORS[method], "Model bu karede zaten \"hortum yok\" diyor; silme testi düşük bir skoru "
                                            "düşürmeye çalıştığı için burada pek bilgi vermez."))
    elif a < ar:
        ratio = ar / max(a, 1e-3)
        r_txt = f"{fmt(ratio, 1)} kat" if ratio < 20 else "20 kattan fazla"
        out.append((P.METHOD_COLORS[method], f"Silme testinde skor, rastgele sıraya göre {r_txt} daha hızlı düşüyor "
                                            f"(alan {fmt(a, 3)} / {fmt(ar, 3)}): {name} modelin gerçekten kullandığı bölgeyi gösteriyor."))
    else:
        out.append((P.METHOD_COLORS[method], f"Silme testinde {name} sırası rastgele sıradan hızlı düşmüyor (alan {fmt(a, 3)} / "
                                            f"{fmt(ar, 3)}): bu karede açıklama modelin kararını taşıyan pikselleri yakalayamıyor."))
    best = min(METHODS, key=lambda k: e["auc"][k])
    if best != method and score >= thr:
        out.append((P.METHOD_COLORS[best], f"Bu karede en sadık açıklama {P.METHOD_NAMES[best]} (alan {fmt(e['auc'][best], 3)})."))
    return out


def _bullets_html(items) -> str:
    return "".join(f"<div class='xb'><i style='background:{c}'></i><span>{t}</span></div>" for c, t in items)


def _frame(ss, store, info):
    if store is None:
        st.html("<div class='note'>Radar verisi bulunamadı; tek kare açıklaması için veri gerekir.</div>")
        return
    if not ss.torch_ok:
        st.html("<div class='note'>Açıklamalar için PyTorch gerekir (<code>pip install torch</code>).</div>")
        return
    variables = list(store.variables)
    if list(info.results.get("variables") or variables) != variables:
        st.html(f"<div class='note'>{info.label} farklı kanallarla eğitilmiş; bu veride açıklanamıyor.</div>")
        return
    rows = _resolve_frame(ss, store)
    r = rows.iloc[ss.ax_frame]
    q = store.images(rows.iloc[[ss.ax_frame]])[0]
    rot = D.rotation_cells(q[None], variables)[0]
    try:
        e = cache.explain(info.key, cache._mtime(info.folder / "best.pt"), str(r["filename"]), q, tuple(variables),
                          None if rot is None else tuple(rot), float(r["az_min"]), float(r["rng_min"]))
    except Exception as ex:  # noqa: BLE001
        st.error(f"Açıklama hesaplanamadı: {ex}")
        return
    method, thr = ss.ax_method, info.threshold
    st.html(_strip(r, ss.ax_ep, e["score"], thr))

    vi = variables.index("VEL") if "VEL" in variables else 0
    left, right = st.columns([2.1, 1], gap="small")
    with left:
        st.html(f"<div class='h-panel'>Model nereye bakıyor? · {P.METHOD_NAMES[method]}</div>")
        tot = np.abs(e["phi"]).sum()
        small = []
        for i, v in enumerate(variables):
            m = e["ig_ch"][i] if method == "ig" else e["maps"][method]
            share = abs(e["phi"][i]) / tot if tot > 1e-6 else 0.0
            small.append((f"{v} · SHAP {e['phi'][i]:+.3f} (%{100 * share:.0f})" + (" · IG bu kanal" if method == "ig" else ""),
                          P.radar_rgb(q[i], v, m)))
        linked.render((variables[vi], P.radar_rgb(q[vi], variables[vi], e["maps"][method])), small, q, variables,
                      e["maps"][method], P.METHOD_NAMES[method], rot=rot, peak=e["peaks"][method],
                      az0=float(r["az_min"]), r0=float(r["rng_min"]))
        st.html(f"<div class='xd'>{METHOD_DESC[method]} Büyük görüntü: {variables[vi]} kanalı, CNN girdisi (120 × 240). "
                "İmleç bütün görüntülerde aynı hücreyi gösterir.</div>")
    with right:
        st.plotly_chart(P.shap_waterfall(e["base"], e["phi"], variables, e["score"], thr, height=360),
                        use_container_width=True, config=CFG)
        st.html("<div class='xd'>Taban: bütün kanallar boşken (yankı yok) skor. Her çubuk bir kanalın kesin Shapley "
                "değeridir (8 ileri geçiş); katkıların toplamı skor − taban farkına tam eşittir.</div>")
        st.plotly_chart(P.deletion_plot(e["deletion"], method, thr, height=330), use_container_width=True, config=CFG)

    d = e["dist_km"][method]
    a, ar = e["auc"][method], e["auc"]["rnd"]
    top = int(np.argmax(np.abs(e["phi"])))
    kpis = [("En etkili kanal", variables[top], f"{e['phi'][top]:+.3f}"),
            ("Zirve ↔ dönme", "—" if d is None else f"{fmt(d, 1)} km", P.METHOD_NAMES[method]),
            ("Silme alanı", fmt(a, 3), f"rastgele {fmt(ar, 3)} · düşük = sadık")]
    st.html("<div class='xbox'><div class='h-panel'>Yorum</div><div class='xk'>"
            + "".join(f"<div><div class='k'>{k}</div><div class='n'>{v}</div><div class='s'>{s}</div></div>" for k, v, s in kpis)
            + "</div>" + _bullets_html(frame_bullets(e, variables, method, thr, rot)) + "</div>")


# ---------------------------------------------------------------- genel

@st.cache_data(show_spinner=False, max_entries=16)
def _load_xai(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def xai_table(info, store) -> pd.DataFrame | None:
    p = info.folder / XAI_FILE
    if not p.exists():
        return None
    df = _load_xai(str(p), cache._mtime(p))
    if store is not None:
        meta = store.samples[["filename", "episode_id", "category", "ef", "split", "end_time"]]
        df = df.merge(meta, on="filename", how="left")
    return df


def shares(df: pd.DataFrame, variables) -> dict:
    """Ortalama |SHAP| payı (her karede kanalların |katkı| toplamı 1'e ölçeklenir)."""
    a = np.abs(df[[f"phi_{v}" for v in variables]].to_numpy())
    s = a.sum(1, keepdims=True)
    a = np.divide(a, s, out=np.zeros_like(a), where=s > 1e-6)
    return {v: float(a[:, i].mean()) for i, v in enumerate(variables)} if len(a) else {}


def global_findings(df: pd.DataFrame, variables, method: str) -> list:
    """Toplu sonuçlardan yorum cümleleri: (renk, metin)."""
    out = []
    by = {k: df[df["kind"] == k] for k in KIND_TR}
    sh = {k: shares(g, variables) for k, g in by.items() if len(g)}
    if "tp" in sh:
        top = max(sh["tp"], key=sh["tp"].get)
        out.append((P.CHANNEL_COLORS.get(top, T.INK), f"Doğru yakalanan hortumlarda en büyük pay {top} kanalının "
                                                      f"(%{100 * sh['tp'][top]:.0f})."))
        if "fn" in sh and "VEL" in sh["tp"]:
            dv = sh["fn"]["VEL"] - sh["tp"]["VEL"]
            if abs(dv) >= 0.03:
                out.append((P.CHANNEL_COLORS["VEL"], f"Kaçırılan hortumlarda VEL'in payı %{100 * sh['fn']['VEL']:.0f}; "
                                                    f"doğru yakalananlara göre {'daha düşük' if dv < 0 else 'daha yüksek'} "
                                                    f"({100 * dv:+.0f} puan)."))
    if "fp" in sh and "tn" in sh:
        v = max(variables, key=lambda c: sh["fp"][c] - sh["tn"][c])
        dv = sh["fp"][v] - sh["tn"][v]
        if dv >= 0.03:
            out.append((P.CHANNEL_COLORS.get(v, T.INK), f"Yanlış alarmlarda {v} kanalının payı doğru negatiflere göre "
                                                        f"{100 * dv:.0f} puan yüksek: yanlış alarmları en çok bu kanaldaki sinyal tetikliyor."))
    col = f"dist_{method}"
    tp = by["tp"][col].dropna() if len(by["tp"]) and col in df else pd.Series(dtype=float)
    fn = by["fn"][col].dropna() if len(by["fn"]) and col in df else pd.Series(dtype=float)
    if len(tp):
        msg = (f"{P.METHOD_NAMES[method]} zirvesinin en güçlü dönmeye {fmt(NEAR_KM, 0)} km'den yakın olduğu kareler: "
               f"doğru pozitiflerde %{100 * (tp <= NEAR_KM).mean():.0f} (ortanca {fmt(tp.median(), 1)} km)")
        if len(fn):
            msg += f", kaçırılanlarda %{100 * (fn <= NEAR_KM).mean():.0f} (ortanca {fmt(fn.median(), 1)} km)."
        else:
            msg += "."
        out.append((T.INK, msg))
    pos = df[df["kind"].isin(["tp", "fp"])]
    if len(pos):
        meths = [k for k in ("cam", "ig", "occ", "loc") if f"auc_{k}" in pos]
        aucs = {k: pos[f"auc_{k}"].mean() for k in (*meths, "rnd")}
        best = min(meths, key=aucs.get)
        worse = [P.METHOD_NAMES[k] for k in meths if aucs[k] >= aucs["rnd"]]
        msg = (f"Sadakat: \"hortum\" denen karelerde en hızlı skor düşüşü {P.METHOD_NAMES[best]} sırasıyla "
               f"(ortalama alan {fmt(aucs[best], 3)}, rastgele {fmt(aucs['rnd'], 3)}).")
        if worse:
            msg += f" {', '.join(worse)} ortalamada rastgeleden iyi değil."
        out.append((P.METHOD_COLORS[best], msg))
    sc = [c for c in ("sanity_cam", "sanity_ig", "sanity_occ", "sanity_loc") if c in df]
    if sc:
        v = {c[7:]: df[c].mean() for c in sc if df[c].notna().any()}
        if v:
            hi = [P.METHOD_NAMES[k] for k, x in v.items() if x > 0.5]
            msg = "Sağlamlık: rastgele ağırlıklı modelin haritası " + ", ".join(
                f"{P.METHOD_NAMES[k]} için ρ = {fmt(x, 2)}" for k, x in v.items()) + "."
            msg += (f" {', '.join(hi)} rastgele modelde de benzer harita veriyor; bu yöntemin haritası kısmen girdinin "
                    "kendi yapısını yansıtıyor." if hi else " Haritalar eğitilmiş modele bağlı.")
            out.append((T.MUTED, msg))
    return out


def _global(ss, store, models, info):
    df = xai_table(info, store)
    if df is None:
        st.html(f"<div class='note'>{info.label} için toplu açıklama sonuçları yok. Bir kez hesapla: "
                f"<code>python scripts/make_xai.py --models {info.key}</code> (önce <code>make_preds.py</code>).</div>")
        return
    variables = [c[4:] for c in df.columns if c.startswith("phi_")]
    method = ss.ax_method
    split = df["split"].dropna().unique().tolist() if "split" in df else []
    n_by = df["kind"].value_counts()

    if f"dist_{method}" not in df:
        st.html(f"<div class='note'>{info.label} için kayıtlı toplu sonuçlarda {P.METHOD_NAMES[method]} yok (eski "
                f"xai.parquet). Yeniden hesapla: <code>python scripts/make_xai.py --models {info.key} --force</code>. "
                "Aşağıda veri seti geneli bölümler yine gösterilir.</div>")
        _dataset_sections(store, info)
        return
    tp = df[df["kind"] == "tp"]
    d = tp[f"dist_{method}"].dropna()
    pos = df[df["kind"].isin(["tp", "fp"])]
    ratio = pos["auc_rnd"].mean() / max(pos[f"auc_{method}"].mean(), 1e-3) if len(pos) else np.nan
    sh_tp = shares(tp, variables)
    top = max(sh_tp, key=sh_tp.get) if sh_tp else "—"
    kpis = [
        (f"Zirve {fmt(NEAR_KM, 0)} km içinde (TP)", f"%{100 * (d <= NEAR_KM).mean():.0f}" if len(d) else "—", f"{len(d)} doğru pozitif kare"),
        ("Ortanca mesafe (TP)", f"{fmt(d.median(), 1)} km" if len(d) else "—", "açıklama zirvesi ↔ en güçlü dönme"),
        ("Rastgeleye göre", f"{fmt(ratio, 1)}×" if ratio == ratio else "—", "silme alanı oranı (hortum denen kareler)"),
        ("Baskın kanal (TP)", top, f"ortalama pay %{100 * sh_tp.get(top, 0):.0f}" if sh_tp else ""),
    ]
    st.html("<div class='xbox'><div class='xk' style='margin:0'>"
            + "".join(f"<div><div class='k'>{k}</div><div class='n'>{v}</div><div class='s'>{s}</div></div>" for k, v, s in kpis)
            + f"</div><div class='xd' style='margin-top:6px'>{P.METHOD_NAMES[method]} · {info.label} · "
            + f"bölüm: {', '.join(split) or '?'} · "
            + ", ".join(f"{KIND_TR[k]} {int(n_by.get(k, 0))}" for k in KIND_TR) + " kare (make_xai.py)</div></div>")

    c1, c2 = st.columns(2, gap="small")
    rows = [(f"{KIND_TR[k]} · {int(n_by.get(k, 0))}", shares(df[df["kind"] == k], variables)) for k in KIND_TR if n_by.get(k, 0)]
    c1.plotly_chart(P.channel_share_bars(rows, variables, height=320), use_container_width=True, config=CFG)
    groups = [(KIND_TR[k], KIND_COLORS[k], df.loc[df["kind"] == k, f"dist_{method}"].dropna()) for k in ("tp", "fn")]
    c2.plotly_chart(P.distance_hist([g for g in groups if len(g[2])], method, height=320), use_container_width=True, config=CFG)

    c3, c4 = st.columns([1.3, 1], gap="small")
    with c3:
        st.html("<div class='h-panel'>Sadakat · silme eğrisi alanı (hortum denen kareler, düşük = sadık)</div>")
        tab = []
        for m in models:
            t = xai_table(m, None)
            if t is None:
                continue
            pp = t[t["kind"].isin(["tp", "fp"])]
            tab.append({"model": m.label, **{P.METHOD_NAMES[k]: round(float(pp[f"auc_{k}"].mean()), 3)
                                             for k in ("cam", "ig", "occ", "loc", "rnd") if f"auc_{k}" in pp},
                        "kare": len(pp)})
        if tab:
            tdf = pd.DataFrame(tab)
            meth_cols = [P.METHOD_NAMES[k] for k in ("cam", "ig", "occ", "loc") if P.METHOD_NAMES[k] in tdf]
            st.dataframe(tdf.style.highlight_min(subset=meth_cols, axis=1, color="#DDF1E8")
                         .format({c: "{:.3f}" for c in [*meth_cols, P.METHOD_NAMES["rnd"]]}),
                         use_container_width=True, hide_index=True)
    with c4:
        sv = {k: float(df[f"sanity_{k}"].mean()) for k in ("cam", "ig", "occ", "loc")
              if f"sanity_{k}" in df and df[f"sanity_{k}"].notna().any()}
        if sv:
            st.plotly_chart(P.sanity_bars(sv, method, height=250), use_container_width=True, config=CFG)

    st.html("<div class='xbox'><div class='h-panel'>Bulgular</div>" + _bullets_html(global_findings(df, variables, method))
            + "<div class='xd'>Cümleler yukarıdaki ölçülerden otomatik üretilir.</div></div>")

    _loc_check(df)
    _dataset_sections(store, info)

    _gallery(ss, store, info, df, method, variables)


def _gallery(ss, store, info, df, method, variables):
    if store is None or not ss.torch_ok:
        return
    c1, c2 = st.columns([1, 3])
    with c1:
        st.html("<div class='h-panel'>Tipik örnekler</div>")
    with c2:
        kind = st.segmented_control("Karar türü", options=["tp", "fn", "fp"], format_func=KIND_TR.get,
                                    default=ss.ax_kind, key="ax_kind_ctl", label_visibility="collapsed")
    if kind:
        ss.ax_kind = kind
    g = df[df["kind"] == ss.ax_kind]
    if g.empty:
        st.html("<div class='note'>Bu türden kare yok.</div>")
        return
    g = g.sort_values("prob", ascending=ss.ax_kind == "fn").head(4)       # en emin olanlar
    cols = st.columns(4, gap="small")
    vi = variables.index("VEL") if "VEL" in variables else 0
    for i, (_, r) in enumerate(g.iterrows()):
        sm = store.samples[store.samples["filename"] == r["filename"]]
        if sm.empty:
            continue
        srow = sm.iloc[0]
        q = store.images(sm)[0]
        rot = D.rotation_cells(q[None], variables)[0]
        try:
            e = cache.explain(info.key, cache._mtime(info.folder / "best.pt"), str(r["filename"]), q, tuple(variables),
                              None if rot is None else tuple(rot), float(srow["az_min"]), float(srow["rng_min"]))
        except Exception as ex:  # noqa: BLE001
            cols[i].error(str(ex))
            continue
        phi = np.asarray(e["phi"])
        tot = np.abs(phi).sum()
        top = int(np.argmax(np.abs(phi)))
        d = e["dist_km"][method]
        with cols[i]:
            st.image(P.radar_rgb(q[vi], variables[vi], e["maps"][method]), use_container_width=True,
                     caption=f"#{srow['episode_id']} · {srow['category']} · skor {fmt(r['prob'], 3)}")
            st.html(f"<div class='xd'>En etkili: <b>{variables[top]}</b> (%{100 * abs(phi[top]) / max(tot, 1e-6):.0f}) · "
                    f"zirve ↔ dönme {'—' if d is None else fmt(d, 1) + ' km'}</div>")
            if st.button("Açıkla", key=f"ax_g_{r['filename']}", icon=":material/lightbulb:", use_container_width=True):
                frames = store.frames(srow["episode_id"])
                open_frame(ss, srow["episode_id"], list(frames.index).index(sm.index[0]), info.key)
                st.rerun()


# ---------------------------------------------------------------- veri seti geneli (X2, X3, X4)

@st.cache_data(show_spinner=False, max_entries=16)
def _parquet(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def _loc_check(df: pd.DataFrame):
    """X4 dürüstlük kontrolleri: haritanın gerçek logit'ten farkı ve sadakatin Grad-CAM ile karşılaştırması."""
    if "loc_gap" not in df or not df["loc_gap"].notna().any():
        return
    g = df["loc_gap"].dropna()
    pos = df[df["kind"].isin(["tp", "fp"])]
    a_loc, a_cam, a_rnd = (pos[f"auc_{k}"].mean() for k in ("loc", "cam", "rnd"))
    verdict = ("Grad-CAM'den sadık" if a_loc < a_cam else "Grad-CAM kadar sadık değil")
    st.html("<div class='xbox'><div class='h-panel'>Konum haritası · dürüstlük kontrolleri (X4)</div>"
            + _bullets_html([
                (P.METHOD_COLORS["loc"], f"Gerçek logit − haritanın en büyüğü: ortanca {fmt(g.median(), 2)}, "
                                         f"|fark| ortancası {fmt(g.abs().median(), 2)} (çeyrekler arası "
                                         f"{fmt(g.quantile(0.25), 2)} … {fmt(g.quantile(0.75), 2)}, {len(g)} kare). "
                                         "Sıfıra yakın değilse harita kararı yalnızca yaklaşık temsil eder."),
                (P.METHOD_COLORS["loc"], f"Silme eğrisi alanı (hortum denen kareler): konum haritası {fmt(a_loc, 3)}, "
                                         f"Grad-CAM {fmt(a_cam, 3)}, rastgele {fmt(a_rnd, 3)} → {verdict}."),
            ]) + "</div>")


def _dataset_sections(store, info):
    if store is None:
        return
    test = store.samples[store.samples["split"] == "test"]
    shp = info.folder / SHAP_FILE
    if shp.exists():
        sh = _parquet(str(shp), cache._mtime(shp)).merge(test[["filename", "category"]], on="filename", how="left")
        variables = [c[4:] for c in sh.columns if c.startswith("phi_")]
        rows = [(f"{CAT_TR.get(c, c)} · {int((sh['category'] == c).sum()):,}".replace(",", "."),
                 shares(sh[sh["category"] == c], variables)) for c in ("TOR", "WRN", "NUL")]
        rows += [(f"{KIND_TR[k]} · {int((sh['kind'] == k).sum()):,}".replace(",", "."), shares(sh[sh["kind"] == k], variables))
                 for k in ("tp", "fn", "fp")]
        st.html("<div class='h-panel'>Kanal katkısı · bütün test kümesi (X3)</div>")
        c1, c2 = st.columns([1.2, 1], gap="small")
        c1.plotly_chart(P.channel_share_bars(rows, variables, height=330, title="Kanal katkısı · kategori ve karar türü"),
                        use_container_width=True, config=CFG)
        tor = shares(sh[sh["category"] == "TOR"], variables)
        c2.plotly_chart(P.channel_vs_auc(tor, EDA_AUC), use_container_width=True, config=CFG)
        order_model = sorted(tor, key=tor.get, reverse=True)
        order_eda = sorted(EDA_AUC, key=EDA_AUC.get, reverse=True)
        same = order_model == [v for v in order_eda if v in tor]
        n_sh = f"{len(sh):,}".replace(",", ".")
        st.html("<div class='note'>Sol: her karede kanalların |SHAP| toplamı 1'e ölçeklenip ortalandı "
                f"({n_sh} kare, 8 ileri geçiş/kare). Sağ: 01 defterinde kanalların tek başına hortumu ayırt etme "
                f"AUC'si (TOR ↔ WRN'de VEL {fmt(EDA_AUC_WRN['VEL'], 3)}, WIDTH {fmt(EDA_AUC_WRN['WIDTH'], 3)}, "
                f"DBZ {fmt(EDA_AUC_WRN['DBZ'], 3)}). Modelin sıralaması ({' > '.join(order_model)}) "
                + ("veri analizindekiyle aynı." if same else f"veri analizindekinden ({' > '.join(order_eda)}) farklı.")
                + "</div>")
    else:
        st.html(f"<div class='note'>Bütün test kümesinde kanal SHAP (X3) ve kompozitler (X1) için bir kez: "
                f"<code>python scripts/make_xai_global.py --models {info.key}</code></div>")

    rot_path = D.npy_dir() / ROT_FILE
    p = cache.probs(info, store.samples)
    if rot_path.exists() and p is not None:
        rot = _parquet(str(rot_path), cache._mtime(rot_path))
        t = test[["filename", "category"]].assign(prob=p[(store.samples["split"] == "test").to_numpy()])
        t = t.merge(rot, on="filename").dropna(subset=["prob", "rot_ms"])
        if len(t) > 20:
            from scipy.stats import spearmanr
            rho = float(spearmanr(t["rot_ms"], t["prob"]).statistic)
            edges = np.unique(np.quantile(t["rot_ms"], np.linspace(0, 1, 11)))
            b = np.clip(np.digitize(t["rot_ms"], edges[1:-1]), 0, len(edges) - 2)
            bins = [(float(t["rot_ms"][b == i].median()), float(t["prob"][b == i].mean()), int((b == i).sum()))
                    for i in range(len(edges) - 1) if (b == i).any()]
            rng = np.random.default_rng(0)
            groups = []
            for c in ("NUL", "WRN", "TOR"):
                g = t[t["category"] == c]
                g = g.iloc[rng.permutation(len(g))[:1500]]
                groups.append((CAT_TR.get(c, c), T.CATEGORY_COLORS[c], g["rot_ms"], g["prob"]))
            st.html("<div class='h-panel'>Model skoru ve dönme şiddeti (X2)</div>")
            st.plotly_chart(P.rotation_scatter(groups, bins, rho, info.threshold), use_container_width=True, config=CFG)
            st.html("<div class='note'>Dönme, VEL'den modelden bağımsız hesaplanır: karenin merkezinde (15° × 15 km) "
                    "5 × 3 hücrelik pencerede en büyük ile en küçük hız farkı (m/s). Noktalar kategori başına en fazla "
                    "1.500 kare; siyah çizgi bütün test karelerinde dönmenin onda birlik aralıklarına göre ortalama "
                    "skor. ρ yüksekse model skoru dönmeyle birlikte artıyor demektir.</div>")
    elif p is not None:
        st.html("<div class='note'>Skor ↔ dönme ilişkisi (X2) için bir kez: "
                "<code>python scripts/make_xai_global.py --only rotation</code></div>")
