"""TorNet hortum tespiti arayüzü (Streamlit).

Çalıştırma (repo kökünden):
    streamlit run src/app/main.py

Veri:    $TORNADO_DATA_DIR/processed/npy  (scripts/export_npy.py ile bir kez hazırlanır)
Modeller: $TORNADO_MODELS_DIR ya da <data_dir>/../models; her alt klasör = bir model (best.pt + results.json)
Skorlar:  models/<ad>/preds.parquet (scripts/make_preds.py)

Sekmeler: Özet · Fırtına Gezgini · Model Karşılaştırma · Hata Analizi.
Üst bar ve Gezgin, src/app/frontend altındaki bileşendir; diğer sekmelerin gövdesi Plotly ile çizilir.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Fırtına Gezgini", page_icon=str(Path(__file__).parent / "assets" / "logo.svg"),
                   layout="wide", initial_sidebar_state="collapsed")

from src.app import cache, shell, theme  # noqa: E402
from src.app import models as M  # noqa: E402
from src.app.tabs import gezgin, hata, kars, ozet  # noqa: E402

TABS = {"ozet": ozet, "gezgin": gezgin, "kars": kars, "hata": hata}

theme.inject()
ss = st.session_state
store, root = cache.store()
models = cache.models()

# ---------------------------------------------------------------- varsayılanlar
ss.setdefault("tab", "gezgin")
ss.setdefault("torch_ok", M.torch_available())
if models and ss.get("sel_model") not in {m.key for m in models}:
    ss.sel_model = models[0].key
ss.setdefault("sel_model", None)
gezgin.defaults(ss)
# splits.csv yoksa veride "val" bölümü olmaz (yalnız katalogdaki train/test); değerlendirme sekmeleri test ile açılsın
if store is not None and not (store.samples["split"] == "val").any():
    for k in ("oz_split", "ks_split", "ht_split"):
        ss.setdefault(k, "test")
for mod in (ozet, kars, hata):
    mod.defaults(ss, models)

# ---------------------------------------------------------------- bileşenden gelen son eylem
ev = ss.get(shell.KEY)
if isinstance(ev, dict) and ev.get("n") != ss.get("_last_ev"):
    ss._last_ev = ev.get("n")
    a = ev.get("a")
    if a == "tab":
        ss.tab = ev["tab"]
    elif a == "model":
        ss.sel_model = ev["key"]
    elif a in ("filter", "split", "goto", "random", "epid"):
        gezgin.handle(ev, ss, store)
    elif a.startswith("ozet_"):
        ozet.handle(ev, ss)
    elif a.startswith("kars_"):
        kars.handle(ev, ss)
    elif a.startswith("hata_"):
        hata.handle(ev, ss)

# ---------------------------------------------------------------- bileşen durumu
public_models = [{"key": m.key, "label": m.label, "long": m.long_label, "color": m.color, "thr": m.threshold,
                  "params": m.params} for m in models]
state = {"tab": ss.tab, "models": public_models, "sel_model": ss.sel_model, "torch": ss.torch_ok}
if ss.tab == "gezgin":
    state["gezgin"] = gezgin.state(ss, store, root, models)
else:
    state[ss.tab] = TABS[ss.tab].state(ss, store, models)

shell.render(state)

if ss.tab != "gezgin":
    if store is None:
        st.html(f"<div class='note'>Radar verisi bulunamadı (<code>{root}</code>). "
                "Ölçütler results.json'dan gösterilir; eğriler için <code>python scripts/export_npy.py</code> ve "
                "<code>python scripts/make_preds.py</code> çalıştır.</div>")
    TABS[ss.tab].body(ss, store, models)
