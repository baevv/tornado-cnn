"""Veri seti geneli açıklanabilirlik ölçüleri (Faz 3, kararlar X1–X3). Önce scripts/make_preds.py çalıştırılmalı.

Model başına (models/<ad>/):
  shap_all.parquet   X3: test kümesinin bütün karelerinde kanal SHAP (DBZ / VEL / WIDTH katkısı, 8 ileri geçiş/kare).
                     Sütunlar: filename, kind (tp / fn / fp / tn), prob, score, base, phi_<kanal>.
  composites.npz     X1: dört uç grubun (en iyi isabet, en kötü yanlış alarm, en kötü ıska, en iyi doğru ret)
                     ortalama radar görüntüsü (fiziksel birim, kanal başına), ortalama dönme haritası (VEL'deki
                     hız farkı; VEL'in kendi ortalaması hız çiftinin yönü değiştiği için sıfırlanır), ortalama
                     Grad-CAM ve ortalama konum haritası (X4). Aynı fırtınadan yalnızca bir kare alınır.
Modelden bağımsız (<npy klasörü>/rotation.parquet):
  X2: her test karesinin merkezindeki en güçlü dönme (m/s, eda.rotation_strength; VEL'deki hız çifti).

Kullanım (repo kökünden):
    python scripts/make_xai_global.py                          # bütün modeller
    python scripts/make_xai_global.py --models 03_baseline --n 75
    python scripts/make_xai_global.py --only composites        # shap | composites | rotation
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import models as M  # noqa: E402
from src.app.data import npy_dir, open_store, physical  # noqa: E402
from src.helper import xai  # noqa: E402

SHAP_FILE, COMP_FILE, ROT_FILE = "shap_all.parquet", "composites.npz", "rotation.parquet"


def _fresh(path: Path, ref: Path) -> bool:
    return path.exists() and path.stat().st_mtime > ref.stat().st_mtime


def rotation(store, split: str, force: bool):
    """X2: modelden bağımsız dönme şiddeti (m/s), bir kez hesaplanır."""
    from src.helper.eda import rotation_strength
    out = npy_dir() / ROT_FILE
    if out.exists() and not force:
        print(f"{ROT_FILE} var, atlanıyor (--force ile yeniden hesaplanır).")
        return
    s = store.samples[store.samples["split"] == split]
    vi = store.variables.index("VEL")
    vals, t0 = np.empty(len(s), dtype=np.float32), time.time()
    for i0 in range(0, len(s), 256):
        rows = s.iloc[i0:i0 + 256]
        vel = physical(store.images(rows), store.variables)[:, vi]
        vals[i0:i0 + len(rows)] = [rotation_strength(v) for v in vel]
        print(f"\r  dönme: {i0 + len(rows):,}/{len(s):,} · {time.time() - t0:.0f} sn", end="", flush=True)
    pd.DataFrame({"filename": s["filename"].to_numpy(), "rot_ms": vals}).to_parquet(out, index=False)
    print(f"\n  → {out}")


def scored(store, info, split):
    p = M.load_preds(info)
    s = store.samples[store.samples["split"] == split].copy()
    s["prob"] = s["filename"].map(p).to_numpy(dtype=np.float32)
    return s[s["prob"].notna()]


def shap_all(store, info, net, s, batch: int):
    """X3: bütün karelerde kanal SHAP."""
    from src.helper.model import bind_range
    v = list(store.variables)
    fire = s["prob"].to_numpy() >= info.threshold
    y = s["label"].to_numpy() == 1
    kind = np.where(y, np.where(fire, "tp", "fn"), np.where(fire, "fp", "tn"))
    parts, t0 = [], time.time()
    for i0 in range(0, len(s), batch):
        rows = s.iloc[i0:i0 + batch]
        n = bind_range(net, rows["rng_min"].to_numpy(dtype=np.float32))
        sh = xai.channel_shapley(n, store.images(rows), v)
        parts.append(pd.DataFrame({"filename": rows["filename"].to_numpy(), "kind": kind[i0:i0 + len(rows)],
                                   "prob": rows["prob"].to_numpy(), "score": sh["score"], "base": sh["base"],
                                   **{f"phi_{c}": sh["phi"][:, j] for j, c in enumerate(v)}}))
        el = time.time() - t0
        done = i0 + len(rows)
        print(f"\r  SHAP: {done:,}/{len(s):,} · {el:.0f} sn · kalan ≈{el / done * (len(s) - done):.0f} sn", end="", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(info.folder / SHAP_FILE, index=False)
    print(f"\n  → {info.folder / SHAP_FILE}")


def composites(store, info, net, s, n: int):
    """X1: dört uç grubun ortalama görüntüsü, Grad-CAM'i ve konum haritası."""
    from src.helper.eda import rotation_map
    from src.helper.model import bind_range
    v = list(store.variables)
    groups = xai.extreme_groups(s, info.threshold, n=n)
    arrays = {"variables": np.array(v)}
    for k, g in groups.items():
        if g.empty:
            continue
        q = store.images(g)
        r0 = g["rng_min"].to_numpy(dtype=np.float32)
        phys = physical(q, v)                                          # (N, C, H, W), boş piksel NaN
        if "VEL" in v:                                                 # hız çiftinin yönü olaydan olaya değişir:
            rot = np.stack([rotation_map(x) for x in phys[:, v.index("VEL")]])   # VEL ortalaması sıfırlanır,
            with np.errstate(all="ignore"):                                     # dönme (hız farkı) ortalaması kalır
                arrays[f"{k}_rot"] = np.nanmean(rot, axis=0).astype(np.float32)
        cams, locs = [], []
        for i0 in range(0, len(q), 32):
            nb = bind_range(net, r0[i0:i0 + 32])
            cams.append(xai.gradcam(nb, q[i0:i0 + 32], v))
            locs.append(xai.location_map(nb, q[i0:i0 + 32], v)["score_map"])
        with np.errstate(all="ignore"):
            arrays[f"{k}_img"] = np.nanmean(phys, axis=0).astype(np.float32)
        arrays[f"{k}_cover"] = np.isfinite(phys).mean(axis=0).astype(np.float32)
        arrays[f"{k}_cam"] = np.concatenate(cams).mean(0).astype(np.float32)
        arrays[f"{k}_loc"] = np.concatenate(locs).mean(0).astype(np.float32)
        arrays[f"{k}_prob"] = g["prob"].to_numpy(dtype=np.float32)
        arrays[f"{k}_files"] = g["filename"].to_numpy().astype(str)
        print(f"  {k}: {len(g)} kare (ortalama skor {g['prob'].mean():.3f})")
    np.savez_compressed(info.folder / COMP_FILE, **arrays)
    print(f"  → {info.folder / COMP_FILE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="model klasör adları (varsayılan: hepsi)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=75, help="kompozit grubu başına kare (X1)")
    ap.add_argument("--only", choices=["shap", "composites", "rotation"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default=None, help="cpu, cuda ya da mps (varsayılan: en hızlısı)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                             else "cpu")
    store = open_store()
    if args.only in (None, "rotation"):
        rotation(store, args.split, args.force)
    if args.only == "rotation":
        return

    infos = M.discover()
    if args.models:
        infos = [m for m in infos if m.key in args.models]
    for info in infos:
        if not info.has_preds:
            print(f"{info.key}: preds.parquet yok, önce scripts/make_preds.py; atlanıyor.")
            continue
        if list(info.results.get("variables") or store.variables) != list(store.variables):
            print(f"{info.key}: kanallar veriyle uyuşmuyor; atlanıyor.")
            continue
        s = scored(store, info, args.split)
        if s.empty:
            print(f"{info.key}: {args.split} bölümünde skor yok; atlanıyor.")
            continue
        best = info.folder / "best.pt"
        todo = [k for k in ("shap", "composites") if args.only in (None, k)]
        todo = [k for k in todo if args.force or not _fresh(info.folder / (SHAP_FILE if k == "shap" else COMP_FILE), best)]
        if not todo:
            print(f"{info.key}: güncel, atlanıyor (--force ile yeniden hesaplanır).")
            continue
        print(f"{info.key}: {len(s):,} kare · {', '.join(todo)} · cihaz {device}")
        net = M.load_model(info, device)
        if "composites" in todo:
            composites(store, info, net, s, args.n)
        if "shap" in todo:
            shap_all(store, info, net, s, args.batch)


if __name__ == "__main__":
    main()
