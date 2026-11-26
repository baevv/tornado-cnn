"""İşlenmiş .npz dosyalarını arayüz için bir kez .npy'ye çevirir ve örnek tablosunu yazar.

Neden: .npz sıkıştırılmış olduğundan açmak bütün yılı belleğe alır (yıl başına birkaç GB).
Sıkıştırmasız .npy ise np.load(mmap_mode="r") ile açılır; yalnız gösterilen kareler diskten okunur.

Kullanım (repo kökünden):
    python scripts/export_npy.py                 # data_dir()/processed içindeki bütün tornet_*.npz
    python scripts/export_npy.py --years 2017    # yalnız bir yıl
    python scripts/export_npy.py --force         # var olan .npy dosyalarının üstüne yaz

Girdi:  <data_dir>/catalog.csv, <data_dir>/splits.csv (varsa), <data_dir>/processed/tornet_<yıl>.npz
Çıktı:  <data_dir>/processed/npy/{tornet_<yıl>_X.npy, samples.parquet, meta.json}
data_dir: $TORNADO_DATA_DIR ya da <repo>/data (bkz. src/helper/paths.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.data import META_FILE, SAMPLES_FILE, npy_dir  # noqa: E402
from src.helper import paths  # noqa: E402
from src.helper.tornet import load_catalog  # noqa: E402

# Katalogdan alınacak sütunlar (yeni ad: katalogdaki ad)
CATALOG_COLS = {"episode_id": "episode_id", "start_time": "start_time", "end_time": "end_time",
                "category": "category", "ef_number": "ef_number", "type": "type", "site": "site_name",
                "lat": "lat", "lon": "lon", "az_min": "az_min", "rng_min": "rng_min"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="*", help="yalnız bu yıllar (varsayılan: bulunan bütün npz)")
    ap.add_argument("--force", action="store_true", help="var olan .npy dosyalarının üstüne yaz")
    args = ap.parse_args()

    data = paths.data_dir()
    proc = paths.processed_dir()
    out = npy_dir()
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(proc.glob("tornet_*.npz"))
    if args.years:
        files = [f for f in files if int(f.stem.split("_")[1]) in args.years]
    if not files:
        sys.exit(f"{proc} içinde tornet_*.npz bulunamadı. TORNADO_DATA_DIR doğru mu?")

    catalog_path = data / "catalog.csv"
    if not catalog_path.exists():
        sys.exit(f"{catalog_path} bulunamadı.")
    cat = load_catalog(catalog_path)
    keep = {k: v for k, v in CATALOG_COLS.items() if v in cat.columns}
    cat = cat[["filename", *keep.values()]].rename(columns={v: k for k, v in keep.items()})

    splits_path = data / "splits.csv"
    split_map = pd.read_csv(splits_path, usecols=["filename", "split"]) if splits_path.exists() else None

    parts, variables = [], None
    for f in files:
        year = int(f.stem.split("_")[1])
        target = out / f"tornet_{year}_X.npy"
        print(f"{f.name} okunuyor...", flush=True)
        with np.load(f) as d:
            names = d["filename"].astype(str)
            y = d["y"].astype(np.uint8)
            v = [str(x) for x in d["variables"]]
            if variables is None:
                variables = v
            elif v != variables:
                sys.exit(f"{f.name}: kanallar {v}, diğer yıllarda {variables}. Aynı olmalı.")
            if target.exists() and not args.force:
                print(f"  {target.name} zaten var, atlanıyor (yenilemek için --force).")
            else:
                X = d["X"]
                tmp = target.with_suffix(".partial.npy")
                np.save(tmp, X)
                tmp.rename(target)
                print(f"  {target.name}: {X.shape}, {target.stat().st_size / 1e9:.2f} GB")
                del X
        parts.append(pd.DataFrame({"year": year, "row": np.arange(len(names)), "filename": names, "label": y}))

    s = pd.concat(parts, ignore_index=True)
    # Sadece --years ile seçilmeyen ama daha önce çıkarılmış yılları da tabloya kat
    for old in sorted(out.glob("tornet_*_X.npy")):
        year = int(old.stem.split("_")[1])
        if year not in set(s["year"]) and (out / SAMPLES_FILE).exists():
            prev = pd.read_parquet(out / SAMPLES_FILE)
            s = pd.concat([s, prev.loc[prev["year"] == year, ["year", "row", "filename", "label"]]], ignore_index=True)

    s = s.merge(cat, on="filename", how="left")
    missing = int(s["start_time"].isna().sum())
    if missing:
        print(f"Uyarı: {missing} örnek katalogda bulunamadı; çıkarıldı.")
        s = s[s["start_time"].notna()].reset_index(drop=True)
    no_ep = s["episode_id"].isna()
    if no_ep.any():                                          # olaysız örnek: kendi başına bir "fırtına"
        s.loc[no_ep, "episode_id"] = -(np.arange(int(no_ep.sum())) + 1)
    s["episode_id"] = s["episode_id"].astype("int64")
    s["ef"] = pd.to_numeric(s.get("ef_number"), errors="coerce").fillna(-1).round().astype(int)
    s.loc[s["category"] != "TOR", "ef"] = -1
    if "rng_min" in s:
        s["rng_min"] = s["rng_min"] / 1000.0                 # metre → km
    if split_map is not None:
        s = s.merge(split_map, on="filename", how="left")
        s["split"] = s["split"].fillna(s["type"])
    else:
        s["split"] = s["type"]                               # bölme yoksa katalogdaki train/test
    s = s.drop(columns=[c for c in ("ef_number",) if c in s]).sort_values(["year", "row"]).reset_index(drop=True)

    s.to_parquet(out / SAMPLES_FILE, index=False)
    (out / META_FILE).write_text(json.dumps({"variables": variables}, ensure_ascii=False), encoding="utf-8")
    print(f"Hazır: {len(s):,} örnek, {s['episode_id'].nunique():,} fırtına → {out}")


if __name__ == "__main__":
    main()
