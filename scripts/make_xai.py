"""Açıklanabilirlik ölçülerini test setinden seçilen karelerde bir kez hesaplar: models/<ad>/xai.parquet.

Arayüzün "Açıklama · Genel" görünümü ve 07 not defteri bu dosyayı okur. Her kare için (src/helper/xai.py):
  kanal SHAP (DBZ/VEL/WIDTH katkısı), açıklama zirvesinin en güçlü dönmeye uzaklığı (km, dört yöntem),
  silme eğrisi alanı (dört yöntem + rastgele sıra), ilk --sanity karede ağırlık rastgeleleştirme testi,
  konum haritasının (X4) gerçek logit'ten farkı (loc_gap).

Kareler karar türüne göre dengeli seçilir (TP, FN, FP, TN; her birinden --n tane, rastgele, seed 0).
Karar türü preds.parquet'teki skordan ve modelin eşiğinden gelir: önce scripts/make_preds.py çalıştırılmalı.

Kullanım (repo kökünden):
    python scripts/make_xai.py                         # bütün modeller, test bölümü, türe göre 50'şer kare
    python scripts/make_xai.py --models 03_baseline --n 100
    python scripts/make_xai.py --split val --force
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import models as M  # noqa: E402
from src.app.data import open_store, rotation_cells  # noqa: E402
from src.helper import xai  # noqa: E402

XAI_FILE = "xai.parquet"
KINDS = ("tp", "fn", "fp", "tn")


def kinds(y: np.ndarray, p: np.ndarray, thr: float) -> np.ndarray:
    fire = p >= thr
    return np.where(y == 1, np.where(fire, "tp", "fn"), np.where(fire, "fp", "tn"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="model klasör adları (varsayılan: hepsi)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=50, help="karar türü başına kare sayısı")
    ap.add_argument("--sanity", type=int, default=20, help="ağırlık rastgeleleştirme testinin yapılacağı kare sayısı")
    ap.add_argument("--device", default=None, help="cpu, cuda ya da mps (varsayılan: en hızlısı)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                             else "cpu")

    store = open_store()
    variables = list(store.variables)
    infos = M.discover()
    if args.models:
        infos = [m for m in infos if m.key in args.models]
    if not infos:
        sys.exit(f"{M.models_root()} altında model bulunamadı.")

    for info in infos:
        out_path = info.folder / XAI_FILE
        if out_path.exists() and not args.force and out_path.stat().st_mtime > (info.folder / "best.pt").stat().st_mtime:
            print(f"{info.key}: {XAI_FILE} güncel, atlanıyor (--force ile yeniden hesaplanır).")
            continue
        if not info.has_preds:
            print(f"{info.key}: preds.parquet yok, önce scripts/make_preds.py; atlanıyor.")
            continue
        if list(info.results.get("variables") or variables) != variables:
            print(f"{info.key}: kanallar veriyle uyuşmuyor; atlanıyor.")
            continue

        p = M.load_preds(info)
        s = store.samples[store.samples["split"] == args.split].copy()
        s["prob"] = s["filename"].map(p).to_numpy(dtype=np.float32)
        s = s[s["prob"].notna()]
        s["kind"] = kinds(s["label"].to_numpy(), s["prob"].to_numpy(), info.threshold)
        rng = np.random.default_rng(0)
        pick = []
        for k in KINDS:
            idx = s.index[s["kind"] == k].to_numpy()
            pick.extend(rng.choice(idx, size=min(args.n, len(idx)), replace=False))
        sel = s.loc[pick]
        print(f"{info.key}: {len(sel)} kare ({', '.join(f'{k} {int((sel.kind == k).sum())}' for k in KINDS)}) · cihaz {device}")

        net = M.load_model(info, device)
        rnd_net = xai.randomized(net) if args.sanity else None
        rows, t0 = [], time.time()
        for n_done, (idx, r) in enumerate(sel.iterrows()):
            q = store.images(r.to_frame().T)[0]
            rot = rotation_cells(q[None], variables)[0]
            e = xai.explain_frame(net, q, variables, rot_cell=rot, az0=float(r["az_min"]), r0=float(r["rng_min"]))
            row = {"filename": r["filename"], "kind": r["kind"], "prob": float(r["prob"]),
                   "score": e["score"], "base": e["base"], "rot_found": rot is not None, "loc_gap": e["loc_gap"],
                   **{f"phi_{v}": float(e["phi"][i]) for i, v in enumerate(variables)},
                   **{f"dist_{k}": (np.nan if d is None else d) for k, d in e["dist_km"].items()},
                   **{f"auc_{k}": a for k, a in e["auc"].items()}}
            if n_done < args.sanity:
                er = xai.explain_frame(rnd_net, q, variables, del_steps=1, n_random=1, r0=float(r["rng_min"]))
                for k in xai.METHODS:
                    row[f"sanity_{k}"] = xai.rank_corr(e["maps"][k], er["maps"][k])
            rows.append(row)
            el = time.time() - t0
            print(f"\r  {n_done + 1}/{len(sel)} · {el:.0f} sn · kalan ≈{el / (n_done + 1) * (len(sel) - n_done - 1):.0f} sn",
                  end="", flush=True)
        out = pd.DataFrame(rows)
        out.to_parquet(out_path, index=False)
        print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
