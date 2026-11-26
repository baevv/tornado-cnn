"""Her modelin bütün örnekler için hortum olasılığını bir kez hesaplar: models/<ad>/preds.parquet.

Arayüz skorları buradan okur; böylece fırtına şeridi, karşılaştırma eğrileri ve hata listesi anında açılır.
Önce scripts/export_npy.py çalıştırılmış olmalı.

Kullanım (repo kökünden):
    python scripts/make_preds.py                       # preds.parquet'i olmayan ya da eskimiş bütün modeller
    python scripts/make_preds.py --models 05_cnn_bn    # yalnız bu model(ler)
    python scripts/make_preds.py --split test val      # yalnız bu bölümler (varsayılan: hepsi)
    python scripts/make_preds.py --force               # hepsini yeniden hesapla

Çıktı sütunları: filename, prob (sigmoid olasılığı).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import models as M  # noqa: E402
from src.app.data import open_store  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="model klasör adları (varsayılan: hepsi)")
    ap.add_argument("--split", nargs="*", help="train / val / test / dropped (varsayılan: hepsi)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None, help="cpu, cuda ya da mps (varsayılan: en hızlısı)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                             else "cpu")

    store = open_store()
    samples = store.samples
    if args.split:
        samples = samples[samples["split"].isin(args.split)]
    print(f"{len(samples):,} örnek · kanallar {store.variables} · cihaz {device}")

    infos = M.discover()
    if args.models:
        infos = [m for m in infos if m.key in args.models]
    if not infos:
        sys.exit(f"{M.models_root()} altında model bulunamadı (best.pt + results.json).")

    for info in infos:
        if info.has_preds and not info.preds_stale and not args.force:
            print(f"{info.key}: preds.parquet güncel, atlanıyor (--force ile yeniden hesaplanır).")
            continue
        variables = info.results.get("variables") or store.variables
        if list(variables) != list(store.variables):
            print(f"{info.key}: model {variables} kanallarıyla eğitilmiş, veri {store.variables}; atlanıyor.")
            continue
        t0 = time.time()
        net = M.load_model(info, device)
        probs = np.empty(len(samples), dtype=np.float32)
        done = 0
        # Yıl yıl, satır sırasıyla okunur: mmap'ten ardışık okuma hızlıdır
        for year, grp in samples.groupby("year", sort=True):
            arr = store.arrays[int(year)]
            pos = np.flatnonzero(samples["year"].to_numpy() == year)
            rows = grp["row"].to_numpy()
            for i in range(0, len(rows), args.batch):
                r = rows[i:i + args.batch]
                probs[pos[i:i + args.batch]] = M.predict_images(net, arr[r], variables, batch=args.batch, device=device)
                done += len(r)
                print(f"\r  {info.key}: {done:,}/{len(samples):,}", end="", flush=True)
        out = pd.DataFrame({"filename": samples["filename"].to_numpy(), "prob": probs})
        if info.has_preds and args.split and not args.force:     # bölüm seçildiyse eskilerle birleştir
            old = pd.read_parquet(info.preds_path)
            out = pd.concat([old[~old["filename"].isin(out["filename"])], out], ignore_index=True)
        out.to_parquet(info.preds_path, index=False)
        print(f"\r  {info.key}: {len(out):,} skor → {info.preds_path} ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
