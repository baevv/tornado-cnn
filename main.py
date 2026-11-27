"""Arayüzü tek komutla başlatır:  python main.py

Sırasıyla:
  1. Veri klasörünü bulur: --data, $TORNADO_DATA_DIR, <repo>/data, Google Drive (~/Library/CloudStorage/GoogleDrive-*/...).
     catalog.csv ile tornet_*.npz aynı klasörde duruyorsa (ör. ~ ya da ~/Downloads) onları <repo>/data'ya bağlar.
  2. Model klasörünü bulur: --models, $TORNADO_MODELS_DIR, <repo>/models, <veri>/../models.
  3. .npy dosyaları yoksa ya da .npz'ler daha yeniyse scripts/export_npy.py'yi çalıştırır
     (yalnız catalog.csv / splits.csv değiştiyse büyük .npy dosyaları yeniden yazılmaz, yalnız örnek tablosu).
  4. Skoru olmayan, eskimiş ya da eksik model varsa scripts/make_preds.py'yi çalıştırır
     (Colab'ın yazdığı preds.parquet yalnız doğrulama + test kapsar; eksik bölümler tamamlanır).
  5. Streamlit'i başlatır ve tarayıcıda açar.
  6. Arka planda, arayüz açıkken açıklanabilirlik hesaplarını yapar (uzun sürer; bittikçe sayfa yenilenince görünür):
       scripts/make_xai_global.py   dönme tablosu (X2), bütün testte kanal SHAP (X3), kompozitler (X1)
       scripts/make_xai.py          kare başına açıklama ölçüleri (konum haritası X4 dahil)

Seçenekler:
  python main.py --data "<veri klasörü>" --models "<model klasörü>"
  python main.py --no-prep        # hazırlığı atla, sadece arayüzü aç
  python main.py --no-xai         # skorları hazırla ama açıklanabilirlik hesaplarını yapma
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


def _is_flat(d: Path) -> bool:
    """Düz düzen: tornet_*.npz ve catalog.csv doğrudan klasörün içinde (processed/ alt klasörü yok)."""
    return (d / "catalog.csv").exists() and any(d.glob("tornet_*.npz"))


def link_flat(src: Path) -> Path:
    """Düz düzendeki dosyaları <repo>/data altına sembolik bağla (kopyalamaz); diğer betikler data/processed bekler."""
    dst = ROOT / "data"
    (dst / "processed").mkdir(parents=True, exist_ok=True)
    pairs = [(src / n, dst / n) for n in ("catalog.csv", "splits.csv")]
    pairs += [(f, dst / "processed" / f.name) for f in src.glob("tornet_*.npz")]
    for s, t in pairs:
        if not s.exists():
            continue
        if t.is_symlink() and t.resolve() != s.resolve():
            t.unlink()                                   # eski bağ başka yeri gösteriyor
        if not t.exists() and not t.is_symlink():
            t.symlink_to(s.resolve())
    print(f"Düz veri klasörü bulundu ({src}); {dst} altına bağlandı.")
    return dst


def find_data(arg):
    """Veri klasörü: içinde processed/tornet_*.npz (ya da hazır npy) olan ilk aday."""
    cands = [arg, os.environ.get("TORNADO_DATA_DIR"), ROOT / "data"]
    home = Path.home()
    for pat in ("Library/CloudStorage/GoogleDrive-*/*/tornado-cnn/data",          # Mac Google Drive
                "Library/CloudStorage/GoogleDrive-*/*/*/tornado-cnn/data",
                "Google Drive/*/tornado-cnn/data", "Google Drive/tornado-cnn/data",
                "Library/CloudStorage/GoogleDrive-*/*/*/data"):
        cands += sorted(glob.glob(str(home / pat)))
    cands = [Path(c).expanduser() for c in cands if c]
    for c in cands:
        if _has_data(c):
            return c
    # catalog.csv ve npz'ler tek klasörde duruyorsa (ör. Colab'dan indirilip ana klasöre konmuşsa)
    given = [Path(c).expanduser() for c in (arg, os.environ.get("TORNADO_DATA_DIR")) if c]
    for c in [*given, home, home / "Downloads", home / "Desktop", home / "Documents"]:
        if c.is_dir() and _is_flat(c):
            return link_flat(c)
    return None


def find_models(arg, data):
    cands = [arg, os.environ.get("TORNADO_MODELS_DIR"), ROOT / "models"]
    if data is not None:
        cands.append(data.parent / "models")
    for c in cands:
        if c and any(Path(c).expanduser().glob("*/best.pt")):
            return Path(c).expanduser()
    return None


def needs_export(data: Path):
    """None: gerek yok · "table": yalnız örnek tablosu (catalog/splits değişti) · "full": .npy'ler de yeniden yazılır."""
    npy_dir = data / "processed" / "npy"
    samples = npy_dir / "samples.parquet"
    npz = list((data / "processed").glob("tornet_*.npz"))
    if not npz:
        return None                                    # sadece npy kopyası varsa yapacak bir şey yok
    if not samples.exists():
        return "full"
    for f in npz:                                      # .npz, kendi .npy'sinden yeniyse (ya da .npy yoksa)
        x = npy_dir / f"{f.stem}_X.npy"
        if not x.exists() or f.stat().st_mtime > x.stat().st_mtime:
            return "full"
    tables = [p for p in (data / "catalog.csv", data / "splits.csv") if p.exists()]
    if tables and max(p.stat().st_mtime for p in tables) > samples.stat().st_mtime:
        return "table"
    return None


def _model_dirs(models: Path):
    return sorted(p for p in models.iterdir() if (p / "best.pt").exists() and (p / "results.json").exists())


def needs_preds(models: Path, data: Path) -> dict:
    """{model: None (hepsi yeniden) | [eksik bölümler]}. Colab'ın preds.parquet'i yalnız val + test kapsar."""
    import pandas as pd
    samples = pd.read_parquet(data / "processed" / "npy" / "samples.parquet", columns=["filename", "split"])
    todo = {}
    for d in _model_dirs(models):
        pr = d / "preds.parquet"
        if not pr.exists() or (d / "best.pt").stat().st_mtime > pr.stat().st_mtime:
            todo[d.name] = None
            continue
        have = set(pd.read_parquet(pr, columns=["filename"])["filename"])
        missing = sorted(samples.loc[~samples["filename"].isin(have), "split"].unique())
        if missing:
            todo[d.name] = missing
    return todo


def _fresh(f: Path, ref: Path) -> bool:
    return f.exists() and f.stat().st_mtime > ref.stat().st_mtime


def needs_xai(models: Path, data: Path) -> tuple[bool, list]:
    """(make_xai_global gerekli mi, make_xai gereken modeller). Skoru olmayan model atlanır."""
    import pyarrow.parquet as pq
    glob_todo = not (data / "processed" / "npy" / "rotation.parquet").exists()
    frame_todo = []
    for d in _model_dirs(models):
        best = d / "best.pt"
        if not (d / "preds.parquet").exists():
            continue
        if not (_fresh(d / "shap_all.parquet", best) and _fresh(d / "composites.npz", best)):
            glob_todo = True
        x = d / "xai.parquet"
        if not _fresh(x, best) or "loc_gap" not in pq.read_schema(x).names:   # eski dosya: konum haritası (X4) yok
            frame_todo.append(d.name)
    return glob_todo, frame_todo


def xai_background(models: Path, data: Path, env):
    """Arayüz açıkken açıklanabilirlik hesapları; çıktı terminale [XAI] önekiyle yazılır."""
    try:
        glob_todo, frame_todo = needs_xai(models, data)
    except Exception as e:  # noqa: BLE001
        print(f"[XAI] kontrol başarısız: {e}", flush=True)
        return
    cmds = []
    if glob_todo:
        cmds.append([sys.executable, "-u", "scripts/make_xai_global.py"])
    if frame_todo:
        cmds.append([sys.executable, "-u", "scripts/make_xai.py", "--force", "--models", *frame_todo])
    if not cmds:
        print("[XAI] açıklanabilirlik sonuçları güncel.", flush=True)
        return
    print("[XAI] arka planda başlıyor (arayüzü kullanabilirsin; bitenler sayfa yenilenince görünür).", flush=True)
    for cmd in cmds:
        name = Path(cmd[2]).name
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = 0.0
        for line in iter(p.stdout.readline, ""):
            for part in line.replace("\r", "\n").splitlines():
                part = part.strip()
                if not part or "Warning" in part:
                    continue
                progress = part[:1].isdigit() or part.startswith(("SHAP:", "dönme:"))
                if progress and time.time() - last < 30:      # ilerleme satırları en fazla 30 sn'de bir
                    continue
                last = time.time() if progress else last
                print(f"[XAI] {part}", flush=True)
        if p.wait():
            print(f"[XAI] {name} başarısız oldu (kod {p.returncode}).", flush=True)
            return
        print(f"[XAI] {name} bitti ({(time.time() - t0) / 60:.1f} dk).", flush=True)
    print("[XAI] bütün açıklanabilirlik hesapları bitti; sayfayı yenile.", flush=True)


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
    ap.add_argument("--no-prep", action="store_true", help="npy/skor/açıklama hazırlığını atla")
    ap.add_argument("--no-xai", action="store_true", help="arka plandaki açıklanabilirlik hesaplarını yapma")
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
        print("Uyarı: veri klasörü bulunamadı (processed/tornet_*.npz ya da catalog.csv + tornet_*.npz yok). Arayüz açılacak ama Gezgin boş olur.\n"
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

    ready = False
    if not args.no_prep and data is not None:
        kind = needs_export(data)
        if kind:
            run([sys.executable, "scripts/export_npy.py", *(["--force"] if kind == "full" else [])], env)
        ready = models is not None and (data / "processed" / "npy" / "samples.parquet").exists()
        if ready:
            todo = needs_preds(models, data)
            full = [k for k, v in todo.items() if v is None]
            if full:
                print(f"Skoru olmayan/eskimiş modeller: {', '.join(full)}")
                run([sys.executable, "scripts/make_preds.py", "--force", "--models", *full], env)
            by_splits = {}
            for k, v in todo.items():
                if v is not None:
                    by_splits.setdefault(tuple(v), []).append(k)
            for splits, keys in by_splits.items():           # Colab skorları: eksik bölümleri tamamla
                print(f"Eksik bölüm skorları ({', '.join(splits)}): {', '.join(keys)}")
                run([sys.executable, "scripts/make_preds.py", "--models", *keys, "--split", *splits], env)
    if ready and not args.no_xai:
        threading.Thread(target=xai_background, args=(models, data, env), daemon=True).start()

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
