"""Ham radar verisinin keşifsel analizi (EDA).

Bir arşivden birkaç bin örnek okuyup her örnek için basit ölçüler hesaplar.
Amaç, "hangi değişkeni, hangi tarama açısını, hangi zaman adımını kullanalım?"
sorularına veriye dayalı cevap vermektir.

Ölçüler görüntünün MERKEZİNDE hesaplanır (15 derece x 15 km), çünkü TorNet
pencereleri olayın konumu ortada olacak şekilde kesilmiştir. Radar gürültüsünü
azaltmak için önce 3x3 medyan filtresi uygulanır.

Her değişken için tek bir özet ölçü; ölçüler fiziksel anlamlarına göre seçilmiştir:
  * DBZ   : 99. yüzdelik (en yüksek yansıtıcılık)   - güçlü fırtınalarda yüksek
  * VEL   : dönme gücü (küçük bir pencerede en büyük hız farkı) - tornado = hız çifti
  * KDP   : 99. yüzdelik                            - yoğun yağış/dolu
  * RHOHV : 1. yüzdelik (en düşük değerler)         - tornado enkazı düşük RHOHV verir
  * ZDR   : 1. yüzdelik (en düşük değerler)         - enkaz ZDR'yi 0'a yaklaştırır
  * WIDTH : 99. yüzdelik                            - türbülans/dönme genişliği artırır
"""
import io
import os

import numpy as np
import pandas as pd

from .tornet import CHANNEL_RANGES, iter_tar_samples

VARIABLES = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]

# Histogram kutuları (değer dağılımı için), her değişkenin olası aralığından biraz geniş
HIST_BINS = {
    "DBZ": np.arange(-35, 81, 1.0),
    "VEL": np.arange(-80, 81, 1.0),
    "KDP": np.arange(-4, 10.1, 0.1),
    "RHOHV": np.arange(0, 1.11, 0.01),
    "ZDR": np.arange(-9, 9.1, 0.1),
    "WIDTH": np.arange(0, 20.1, 0.25),
}


# Görüntünün merkezi: 120 açı hücresinin ortadaki 30'u (15 derece), 240 mesafe hücresinin ortadaki 60'ı (15 km)
CENTER = (slice(45, 75), slice(90, 150))


def denoise(arr: np.ndarray) -> np.ndarray:
    """3x3 medyan filtresi: tek tük gürültülü pikselleri bastırır, boş pikseller boş kalır."""
    from scipy.ndimage import median_filter
    out = median_filter(np.nan_to_num(arr, nan=0.0), size=3)
    out[np.isnan(arr)] = np.nan
    return out


def rotation_map(vel: np.ndarray) -> np.ndarray:
    """Her piksel çevresinde (5 açı x 3 mesafe hücresi) en büyük ile en küçük hız arasındaki fark (m/s).

    Tornado radarda yan yana duran yaklaşan (-) ve uzaklaşan (+) rüzgar olarak
    görünür (hız çifti); bu fark ne kadar büyükse dönme o kadar güçlüdür.
    vel: (azimuth, range).
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    v = denoise(vel)
    hi = maximum_filter(np.nan_to_num(v, nan=-999.0), size=(5, 3))
    lo = minimum_filter(np.nan_to_num(v, nan=999.0), size=(5, 3))
    diff = hi - lo
    diff[(hi < -900) | (lo > 900)] = np.nan      # çevresinde hiç ölçüm olmayan pikseller
    return diff


def rotation_strength(vel: np.ndarray, center=True) -> float:
    """Görüntünün merkezindeki en güçlü dönme (m/s)."""
    m = rotation_map(vel)
    if center:
        m = m[CENTER]
    return float(np.nanmax(m)) if np.isfinite(m).any() else np.nan


def rotation_location(vel: np.ndarray, center=True):
    """En güçlü dönmenin (azimuth, range) indisi. Görselleştirmede daire çizmek için."""
    m = rotation_map(vel)
    if center:
        sub = m[CENTER]
        if not np.isfinite(sub).any():
            return None
        a, r = np.unravel_index(np.nanargmax(sub), sub.shape)
        return int(a + CENTER[0].start), int(r + CENTER[1].start)
    if not np.isfinite(m).any():
        return None
    a, r = np.unravel_index(np.nanargmax(m), m.shape)
    return int(a), int(r)


def _summary(v: str, arr: np.ndarray) -> float:
    """Bir değişkenin görüntüsünü tek bir sayıya indirir (dosyanın başındaki tabloya göre).

    VEL için merkezdeki dönme gücü; diğerleri için gürültüsü temizlenmiş merkezde
    1. ya da 99. yüzdelik.
    """
    if v == "VEL":
        return rotation_strength(arr)
    c = denoise(arr)[CENTER]
    ok = c[np.isfinite(c)]
    if ok.size == 0:
        return np.nan
    if v in ("RHOHV", "ZDR"):
        return float(np.percentile(ok, 1))
    return float(np.percentile(ok, 99))


def sample_features(raw: bytes, frame=-1):
    """Bir örnek için: her değişken ve tarama açısında boş piksel oranı ve özet ölçü.

    Ayrıca sweep 0'daki değerleri histogram için ve frame_labels'ı döner.
    """
    import h5py
    feats, values = {}, {}
    with h5py.File(io.BytesIO(raw), "r") as h:
        for v in VARIABLES:
            d = h[v]
            fill = d.attrs.get("_FillValue")
            for s in (0, 1):
                arr = d[frame, :, :, s].astype(np.float32)
                if fill is not None:
                    arr[arr == fill[0]] = np.nan
                feats[f"{v}_s{s}_nan"] = float(np.isnan(arr).mean())
                feats[f"{v}_s{s}"] = _summary(v, arr)
                if s == 0:
                    values[v] = arr[np.isfinite(arr)]
                    if v == "VEL":
                        loc = rotation_location(arr, center=False)
                        feats["rot_az"], feats["rot_rng"] = loc if loc else (np.nan, np.nan)
        labels = h["frame_labels"][:].astype(int)
    return feats, values, labels


def scan_archive(tar_path_or_url, catalog: pd.DataFrame, n_files=1500, keep_raw=("TOR",),
                 keep_other=2, progress=True):
    """Bir arşivin başından n_files örnek okuyup EDA için gereken her şeyi toplar.

    Döner: dict
      features   - DataFrame, her satır bir örnek (+ katalogdan category, ef_number, label)
      hists      - {değişken: histogram sayıları} (sweep 0, tüm örnekler; kutular HIST_BINS'te)
      outside    - {kanal: [ölçekleme aralığı dışında kalan piksel sayısı, toplam piksel]}
      steps      - {kanal: ardışık farklı değerler arasındaki adımlar} -> ölçüm hassasiyeti
      raw        - {dosya_adı: bytes}: tüm TOR örnekleri + diğer kategorilerden birkaç tane
                   (görselleştirme için; tüm örnekleri saklamak bellek doldururdu)
    """
    meta = catalog.set_index("filename")
    rows = []
    hists = {v: np.zeros(len(HIST_BINS[v]) - 1, dtype=np.int64) for v in VARIABLES}
    used = ("DBZ", "VEL", "WIDTH")                   # modelde kullanılan kanallar
    outside = {v: [0, 0] for v in used}             # [aralık dışı piksel, toplam piksel]
    uniques = {v: set() for v in used}
    raw_keep, other_counts = {}, {}

    it = iter_tar_samples(tar_path_or_url)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, total=n_files, desc="örnek okunuyor", unit="dosya")
    for i, (name, raw) in enumerate(it):
        feats, values, labels = sample_features(raw)
        cat = os.path.basename(name)[:3]
        feats.update(filename=name, category=cat,
                     ef_number=meta.at[name, "ef_number"] if name in meta.index else np.nan,
                     first_tor_frame=int(np.argmax(labels)) if labels.any() else -1,
                     n_tor_frames=int(labels.sum()), last_label=int(labels[-1]))
        rows.append(feats)
        for v in VARIABLES:
            hists[v] += np.histogram(values[v], bins=HIST_BINS[v])[0]
        for v in used:
            lo, hi = CHANNEL_RANGES[v]
            outside[v][0] += int(((values[v] < lo) | (values[v] > hi)).sum())
            outside[v][1] += values[v].size
            if len(uniques[v]) < 5000:
                uniques[v].update(np.unique(values[v]).tolist())
        if cat in keep_raw:
            raw_keep[name] = raw
        elif other_counts.get(cat, 0) < keep_other:
            raw_keep[name] = raw
            other_counts[cat] = other_counts.get(cat, 0) + 1
        if i + 1 >= n_files:
            break

    df = pd.DataFrame(rows)
    df["label"] = (df["category"] == "TOR").astype(int)
    steps = {}
    for v, u in uniques.items():
        d = np.diff(np.sort(np.fromiter(u, dtype=np.float64)))
        steps[v] = pd.Series(np.round(d, 4)).value_counts().head(5)
    return {"features": df, "hists": hists, "outside": outside, "steps": steps, "raw": raw_keep}


def separability(df: pd.DataFrame, columns, negative=("WRN", "NUL")):
    """Her ölçünün tornadoyu ne kadar ayırt ettiği: AUC (0,5 = tahminden farksız, 1 = mükemmel).

    AUC < 0,5 ise ölçü ters yönde ayırt ediyor demektir (tornadoda daha DÜŞÜK);
    o yüzden 'ayırt etme gücü' = max(AUC, 1 - AUC) de verilir.
    """
    from sklearn.metrics import roc_auc_score
    sub = df[df["category"].isin(("TOR",) + tuple(negative))]
    y = (sub["category"] == "TOR").astype(int)
    out = []
    for c in columns:
        x = sub[c]
        ok = x.notna()
        auc = roc_auc_score(y[ok], x[ok]) if y[ok].nunique() == 2 else np.nan
        out.append({"ölçü": c, "AUC": auc, "ayırt etme gücü": max(auc, 1 - auc)})
    return pd.DataFrame(out).set_index("ölçü").round(3)
