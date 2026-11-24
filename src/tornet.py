"""TorNet: indirme, katalog, tek örnek okuma ve ön işleme.

Veri seti hakkında bilgiler (v1.1 dosyalarından kontrol edildi):
  * Bir örnek = bir NetCDF (.nc) dosyası.
  * Her radar değişkeninin boyutu (time=4, azimuth=120, range=240, sweep=2):
    5 dakika arayla 4 zaman adımı; 0,5 derece / 250 m çözünürlükte 60 derece x 60 km'lik pencere;
    0,5 ve 0,9 derecelik iki tarama açısı.
  * Ölçüm yapılamayan pikseller NaN.
  * `frame_labels` her kare için bir etiket verir. Örneğin etiketi SON karenin etiketidir,
    ve bu tam olarak category == "TOR" olduğunda 1'dir.
  * catalog.csv tüm yıllardaki 203.133 örneğin listesidir.

Kaynak: Veillette ve ark. (2025), "A Benchmark Dataset for Tornado Detection and
Prediction using Full-Resolution Polarimetric Weather Radar Data",
AIES 4(1). Veri: Zenodo, CC BY 4.0.
"""
import io
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Her yılın Zenodo kayıt numarası (veri seti sürüm 1.1). 2013 kaydında catalog.csv de var.
ZENODO_RECORDS = {
    2013: 12636522, 2014: 12637032, 2015: 12655151, 2016: 12655179, 2017: 12655183,
    2018: 12655187, 2019: 12655716, 2020: 12655717, 2021: 12655718, 2022: 12655719,
}

# Ölçekleme için sabit fiziksel aralıklar (TorNet'in referans kodundan; WIDTH aralığı EDA'ya göre).
# Verimizin istatistikleri değil sabit aralıklar kullanıldığı için test kümesinden eğitime bilgi sızmaz.
CHANNEL_RANGES = {
    "DBZ": (-20.0, 60.0),    # yansıtıcılık, dBZ
    "VEL": (-60.0, 60.0),    # radyal hız, m/s
    "KDP": (-2.0, 5.0),
    "RHOHV": (0.2, 1.04),
    "ZDR": (-1.0, 8.0),
    "WIDTH": (0.0, 16.0),    # hız dağınıklığı (spectrum width), m/s. Üst sınır EDA'ya göre seçildi:
}                            # 0–16 piksellerin %99,9'unu kapsıyor (01 · D2).

CATEGORY_NAMES = {
    "TOR": "doğrulanmış hortum",
    "WRN": "hortum uyarısı verilmiş, hortum yok",
    "NUL": "rastgele fırtına, hortum yok",
}


def file_url(year: int, filename: str) -> str:
    """Zenodo'daki bir dosyanın indirme adresi."""
    return f"https://zenodo.org/records/{ZENODO_RECORDS[year]}/files/{filename}?download=1"


def tar_url(year: int) -> str:
    """Bir yılın arşivinin (tornet_<yıl>.tar.gz) indirme adresi."""
    return file_url(year, f"tornet_{year}.tar.gz")


def remote_size(year: int, filename: str):
    """Zenodo'daki dosyanın bayt cinsinden boyutu (indirmenin tamamlandığını kontrol etmek için)."""
    import json
    try:
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{ZENODO_RECORDS[year]}", timeout=30) as r:
            files = json.load(r)["files"]
        return next(f["size"] for f in files if f["key"] == filename)
    except Exception:
        return None                       # boyut alınamazsa kontrol atlanır


def download(url: str, dest: Path, expected_size=None, max_attempts=30) -> Path:
    """Dosyayı indirir; bağlantı koparsa kaldığı yerden devam ederek tekrar dener.

    Büyük dosyalarda sunucu bağlantıyı yarıda kesebilir (curl hata kodu 18). Her denemede
    curl -C - ile inmiş kısım korunur ve sadece kalan kısım indirilir. expected_size
    verilirse dosya o boyuta ulaşana kadar denemeye devam eder; dosya zaten tamsa hiç indirmez.
    """
    import time

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def done():
        return expected_size is not None and dest.exists() and dest.stat().st_size == expected_size

    if done():
        print(f"  {dest.name} zaten tamamen inmiş.")
        return dest
    if not shutil.which("curl"):          # curl yoksa: devam etme desteği olmadan indir
        urllib.request.urlretrieve(url, dest)
        return dest

    for attempt in range(1, max_attempts + 1):
        res = subprocess.run(["curl", "-L", "-sS", "-C", "-", "--fail", "--retry", "5", "--retry-delay", "5",
                              "--connect-timeout", "30", "-o", str(dest), url])
        have = dest.stat().st_size / 1e9 if dest.exists() else 0.0
        if done() or (expected_size is None and res.returncode == 0):
            print(f"  indirme tamam: {have:.2f} GB")
            return dest
        total = f" / {expected_size / 1e9:.2f}" if expected_size else ""
        print(f"  bağlantı kesildi (deneme {attempt}, curl kodu {res.returncode}); "
              f"inen: {have:.2f}{total} GB. Kaldığı yerden devam ediliyor...")
        time.sleep(5)
    raise RuntimeError(f"{dest.name} {max_attempts} denemede indirilemedi. Hücreyi tekrar çalıştırınca kaldığı yerden devam eder.")


# ---------------------------------------------------------------- katalog

def load_catalog(path) -> pd.DataFrame:
    """catalog.csv'yi okur, `year` (yıl) ve ikili `label` (TOR=1, WRN/NUL=0) sütunlarını ekler."""
    cat = pd.read_csv(path, parse_dates=["start_time", "end_time"])
    cat["year"] = cat["start_time"].dt.year
    cat["label"] = (cat["category"] == "TOR").astype(np.uint8)
    return cat


def get_catalog(data_dir) -> pd.DataFrame:
    """Katalog klasörde yoksa önce indirir, sonra okur."""
    path = Path(data_dir) / "catalog.csv"
    if not path.exists():
        download(file_url(2013, "catalog.csv"), path)
    return load_catalog(path)


# ---------------------------------------------------------------- tek örnek

def open_sample(source):
    """Bir .nc örneğini dosya yolundan ya da ham baytlardan açar (xarray Dataset döner)."""
    import xarray as xr
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    return xr.load_dataset(source, engine="h5netcdf")


def to_uint8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fiziksel değerleri 1..255 aralığına ölçekler; 0'ı 'boş piksel' (NaN) için ayırır.

    Piksel başına 4 bayt yerine 1 bayt kullanmak işlenmiş veriyi 4 kat küçültür.
    Kaybedilen hassasiyet: 80 dBZ / 254 adım = 0,31 dBZ; 120 m/s / 254 = 0,47 m/s; 16 m/s / 254 = 0,06 m/s.
    """
    out = np.zeros(x.shape, dtype=np.uint8)
    ok = np.isfinite(x)                                   # NaN olmayan pikseller
    scaled = (np.clip(x[ok], lo, hi) - lo) / (hi - lo)    # aralık dışını kırp, 0..1'e ölçekle
    out[ok] = np.round(scaled * 254).astype(np.uint8) + 1
    return out


def from_uint8(q: np.ndarray, variables) -> np.ndarray:
    """Saklanan baytları [-1, 1] aralığında ağ girdisine geri çevirir.

    Boş pikseller (0), ölçeklemeden sonra körü körüne 0 yapılmak yerine
    fiziksel olarak anlamlı bir değerle doldurulur:
      * DBZ:   -1 (= en düşük yansıtıcılık, yani "yankı yok")
      * VEL:    0 (= sıfır radyal hız, yani "hareket yok")
      * WIDTH: -1 (= en düşük hız dağınıklığı, yani "türbülans yok")
    """
    x = (q.astype(np.float32) - 1.0) / 254.0 * 2.0 - 1.0   # 1..255 -> -1..1
    missing = q == 0
    fill_values = {"DBZ": -1.0, "VEL": 0.0, "WIDTH": -1.0}
    ch_axis = -3  # boyut sırası (..., kanal, yükseklik, genişlik)
    for i, v in enumerate(variables):
        sl = [slice(None)] * x.ndim
        sl[ch_axis] = i
        sl = tuple(sl)
        x[sl][missing[sl]] = fill_values.get(v, -1.0)
    return x


def preprocess_sample(ds, variables=("DBZ", "VEL"), sweep=0, frame=-1) -> np.ndarray:
    """Kanalları, tarama açısını ve zaman adımını seçer -> (C, 120, 240) boyutunda uint8 dizi."""
    chans = []
    for v in variables:
        lo, hi = CHANNEL_RANGES[v]
        arr = ds[v].values[frame, :, :, sweep]  # (açı, mesafe)
        chans.append(to_uint8(arr, lo, hi))
    return np.stack(chans)


def preprocess_bytes(raw: bytes, variables=("DBZ", "VEL"), sweep=0, frame=-1):
    """preprocess_sample(open_sample(raw)) ile aynı sonucu verir, yaklaşık 10 kat daha hızlı.

    Dosyanın tamamını xarray ile yüklemek yerine h5py ile sadece gereken dilimleri okur.
    Bütün bir yılı işlerken kullanılır. Döner: (uint8 dizi, etiket).
    """
    import h5py
    with h5py.File(io.BytesIO(raw), "r") as h:
        chans = []
        for v in variables:
            d = h[v]
            arr = d[frame, :, :, sweep].astype(np.float32)
            fill = d.attrs.get("_FillValue")         # dosyada boş pikseli işaretleyen değer (-999)
            if fill is not None:
                arr[arr == fill[0]] = np.nan
            chans.append(to_uint8(arr, *CHANNEL_RANGES[v]))
        label = int(h["frame_labels"][frame])
    return np.stack(chans), label


# ---------------------------------------------------------------- bütün bir yıl

def iter_tar_samples(tar_path_or_url):
    """Bir TorNet tar.gz arşivindeki her .nc için (dosya_adı, baytlar) üretir.

    Arşiv diske açılmaz; dosyalar sırayla bellekte okunur.
    """
    src = str(tar_path_or_url)
    fileobj = urllib.request.urlopen(src) if src.startswith("http") else open(src, "rb")
    with fileobj, tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".nc"):
                yield member.name, tar.extractfile(member).read()


def sample_from_tar(tar_path_or_url, per_category=None, max_files=5000):
    """Arşivin başını okuyarak her kategoriden birkaç örnek alır.

    Yeterli örnek bulunur bulunmaz okuma durur; böylece 15 GB'lık arşivin
    sadece küçük bir kısmı indirilir. Döner: {dosya_adı: xarray.Dataset}.
    """
    import os
    per_category = per_category or {"TOR": 2, "WRN": 1, "NUL": 1}
    found, counts = {}, {k: 0 for k in per_category}
    for i, (name, raw) in enumerate(iter_tar_samples(tar_path_or_url)):
        cat = os.path.basename(name)[:3]   # dosya adının ilk 3 harfi kategoriyi verir
        if counts.get(cat, 0) < per_category.get(cat, 0):
            found[name] = open_sample(raw)
            counts[cat] += 1
        if all(counts[k] >= v for k, v in per_category.items()) or i + 1 >= max_files:
            break
    return found


def process_year(year, tar_path_or_url, out_dir, variables=("DBZ", "VEL"), sweep=0, frame=-1,
                 catalog=None, progress=True) -> Path:
    """Bir yılın arşivini X, y ve dosya adlarını içeren tek bir sıkıştırılmış .npz dosyasına çevirir.

    Çıktı: <out_dir>/tornet_<yıl>.npz
      X        uint8 (N, C, 120, 240)   - bkz. to_uint8 / from_uint8
      y        uint8 (N,)               - 1 = hortum
      filename str   (N,)               - catalog.csv ile eşleştirip bölme/fırtına/EF bilgisini almak için
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tornet_{year}.npz"
    if out_path.exists():
        print(f"{out_path.name} zaten var, atlanıyor.")
        return out_path

    expected = None                       # ilerleme çubuğu için beklenen dosya sayısı
    if catalog is not None:
        expected = int((catalog["year"] == year).sum())

    X, y, names = [], [], []
    it = iter_tar_samples(tar_path_or_url)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, total=expected, desc=f"TorNet {year}", unit="dosya")
    for name, raw in it:
        x, label = preprocess_bytes(raw, variables, sweep, frame)
        X.append(x)
        y.append(label)
        names.append(name)

    X = np.stack(X)
    y = np.asarray(y, dtype=np.uint8)
    names = np.asarray(names)
    tmp = out_path.with_name(out_path.stem + ".partial.npz")
    np.savez_compressed(tmp, X=X, y=y, filename=names, variables=np.asarray(variables))
    tmp.rename(out_path)  # son adı sadece tamamlanmış dosya alır (yarım dosya "bitti" sanılmasın)
    print(f"Kaydedildi: {out_path} | {len(y)} örnek, {int(y.sum())} hortum | "
          f"{out_path.stat().st_size / 1e9:.2f} GB")
    return out_path
