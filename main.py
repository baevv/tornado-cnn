"""Arayüzü tek komutla başlatır:  python main.py

Sırasıyla:
  1. Veri klasörünü bulur: --data, $TORNADO_DATA_DIR, <repo>/data, Google Drive (~/Library/CloudStorage/GoogleDrive-*/...).
  2. Model klasörünü bulur: --models, $TORNADO_MODELS_DIR, <repo>/models, <veri>/../models.
  3. .npy dosyaları yoksa ya da .npz'ler daha yeniyse scripts/export_npy.py'yi çalıştırır.
  4. preds.parquet'i olmayan ya da eskimiş model varsa scripts/make_preds.py'yi çalıştırır.
  5. Streamlit'i başlatır ve tarayıcıda açar.

Seçenekler:
  python main.py --data "<veri klasörü>" --models "<model klasörü>"
  python main.py --no-prep        # hazırlığı atla, sadece arayüzü aç
  python main.py --port 8502
"""
import argparse
import glob
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _has_data(d: Path) -> bool:
    return (d / "processed" / "npy" / "samples.parquet").exists() or any((d / "processed").glob("tornet_*.npz"))


def find_data(arg):
    """Veri klasörü: içinde processed/tornet_*.npz (ya da hazır npy) olan ilk aday."""
    cands = [arg, os.environ.get("TORNADO_DATA_DIR"), ROOT / "data"]
    home = Path.home()
    for pat in ("Library/CloudStorage/GoogleDrive-*/*/tornado-cnn/data",          # Mac Google Drive
                "Library/CloudStorage/GoogleDrive-*/*/*/tornado-cnn/data",
                "Google Drive/*/tornado-cnn/data", "Google Drive/tornado-cnn/data",
                "Library/CloudStorage/GoogleDrive-*/*/*/data"):
        cands += sorted(glob.glob(str(home / pat)))
    for c in cands:
        if c and _has_data(Path(c).expanduser()):
            return Path(c).expanduser()
    return None


def find_models(arg, data):
    cands = [arg, os.environ.get("TORNADO_MODELS_DIR"), ROOT / "models"]
    if data is not None:
        cands.append(data.parent / "models")
    for c in cands:
        if c and any(Path(c).expanduser().glob("*/best.pt")):
            return Path(c).expanduser()
    return None


def needs_export(data: Path) -> bool:
    npy = data / "processed" / "npy" / "samples.parquet"
    npz = list((data / "processed").glob("tornet_*.npz"))
    if not npz:
        return False                                   # sadece npy kopyası varsa yapacak bir şey yok
    if not npy.exists():
        return True
    newest = max(p.stat().st_mtime for p in [*npz, data / "catalog.csv", data / "splits.csv"] if p.exists())
    return newest > npy.stat().st_mtime


def needs_preds(models: Path) -> list:
    stale = []
    for d in sorted(p for p in models.iterdir() if (p / "best.pt").exists() and (p / "results.json").exists()):
        pr = d / "preds.parquet"
        if not pr.exists() or (d / "best.pt").stat().st_mtime > pr.stat().st_mtime:
            stale.append(d.name)
    return stale


def run(cmd, env):
    print("→", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode:
        sys.exit(f"Komut başarısız oldu (kod {r.returncode}). Yukarıdaki hatayı kontrol et.")


def open_when_ready(url, timeout=60):
    """Sunucu cevap verince varsayılan tarayıcıda açar."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=1)
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="veri klasörü (catalog.csv, splits.csv, processed/)")
    ap.add_argument("--models", help="model klasörü (her alt klasörde best.pt + results.json)")
    ap.add_argument("--no-prep", action="store_true", help="npy/skor hazırlığını atla")
    ap.add_argument("--port", default="8501")
    args = ap.parse_args()

    try:
        import streamlit  # noqa: F401
        import plotly  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as e:
        sys.exit(f"Eksik paket: {e.name}. Önce:  pip install -r requirements.txt")

    env = os.environ.copy()
    data = find_data(args.data)
    if data is None:
        print("Uyarı: veri klasörü bulunamadı (processed/tornet_*.npz yok). Arayüz açılacak ama Gezgin boş olur.\n"
              "  Klasörü göster:  python main.py --data \"<.../tornado-cnn/data>\"")
    else:
        env["TORNADO_DATA_DIR"] = str(data)
        print(f"Veri:     {data}")
    models = find_models(args.models, data)
    if models is None:
        print("Uyarı: model klasörü bulunamadı (*/best.pt yok). Göster:  python main.py --models \"<klasör>\"")
    else:
        env["TORNADO_MODELS_DIR"] = str(models)
        print(f"Modeller: {models}")

    if not args.no_prep and data is not None:
        if needs_export(data):
            run([sys.executable, "scripts/export_npy.py", "--force"], env)
        if models is not None and (data / "processed" / "npy" / "samples.parquet").exists():
            stale = needs_preds(models)
            if stale:
                print(f"Skoru eksik/eskimiş modeller: {', '.join(stale)}")
                run([sys.executable, "scripts/make_preds.py", "--models", *stale], env)

    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "app" / "main.py"),
           "--server.port", str(args.port), "--server.headless", "true"]   # headless: ilk açılıştaki e-posta sorusu çıkmaz
    url = f"http://localhost:{args.port}"
    print(f"→ arayüz açılıyor: {url}  (kapatmak için Ctrl+C)", flush=True)
    threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    try:
        subprocess.run(cmd, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
