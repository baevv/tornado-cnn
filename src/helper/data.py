"""Veriyi eğitim / doğrulama / test kümelerine bölme ve bölmenin kontrolleri.

Kurallar:
  * Test kümesi: veri setinin hazır `test` kısmı. Hiç değiştirilmez, en sona kadar kullanılmaz.
  * Test fırtınalarıyla ortak olan fırtınalar (episode_id) eğitimden çıkarılır (sızıntı).
  * Doğrulama kümesi: kalan `train` kısmından, FIRTINA OLAYINA GÖRE ayrılır. Aynı fırtınanın
    bütün örnekleri ya eğitimde ya doğrulamada olur.
  * Doğrulamada hortum oranı eğitimdekine yakın olsun diye bölme etikete göre dengelenir.
  * Farklı fırtına bölümünde olsa da bir doğrulama örneğine zamanda ve konumda çok yakın
    (30 dk, 0,25°) eğitim örnekleri çıkarılır; doğrulama örneklerinin %1'inden fazlasında böyle
    bir komşu varsa (karar P5, reports/kararlar.md).
"""
import numpy as np
import pandas as pd

SPLIT_COLUMNS = ["filename", "year", "type", "category", "label", "ef_number", "episode_id", "rng_min", "split"]
# rng_min (metre): CoordConv kanalları (r, 1/r) için ilk mesafe hücresinin radara uzaklığı (karar K4 / P2)


def make_splits(catalog: pd.DataFrame, years, val_fraction=0.15, seed=42,
                neighbor_threshold_pct=1.0) -> pd.DataFrame:
    """Seçilen yılların her örneğine bir küme atar: 'train', 'val', 'test' ya da 'dropped'.

    'dropped' = eğitimden çıkarılan örnek. İki nedeni olabilir:
      * test kümesindeki bir fırtınaya (episode_id) ait,
      * bir doğrulama örneğine zamanda ve konumda çok yakın (30 dk, 0,25°; bkz. drop_val_neighbors).
        Bu kural yalnızca doğrulama örneklerinin %neighbor_threshold_pct'sinden fazlasının eğitimde
        komşusu varsa uygulanır (karar P5). None verilirse uygulanmaz.
    Aynı seed ile her çalıştırmada aynı bölme çıkar.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    df = catalog[catalog["year"].isin(years)].copy()
    df["split"] = np.where(df["type"] == "test", "test", "train")

    # 1) Test fırtınalarıyla ortak fırtınaları eğitimden çıkar
    test_eps = set(df.loc[df["split"] == "test", "episode_id"])
    overlap = (df["split"] == "train") & df["episode_id"].isin(test_eps)
    df.loc[overlap, "split"] = "dropped"

    # 2) Kalan eğitim kısmından doğrulamayı fırtınaya göre ayır.
    #    StratifiedGroupKFold: gruplar (fırtınalar) bölünmez, etiket oranı korunmaya çalışılır.
    #    1 / n_splits kadarı doğrulamaya gider.
    pool = df[df["split"] == "train"]
    n_splits = max(2, int(round(1 / val_fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(pool, pool["label"], groups=pool["episode_id"]))
    df.loc[pool.index[val_idx], "split"] = "val"
    out = df[SPLIT_COLUMNS].reset_index(drop=True)

    # 3) Doğrulama örneklerine zamanda ve konumda komşu eğitim örneklerini çıkar (P5, %1 kuralı)
    if neighbor_threshold_pct is not None:
        out, _ = drop_val_neighbors(out, catalog, threshold_pct=neighbor_threshold_pct)
    return out


def neighbor_pairs(catalog: pd.DataFrame, a: pd.Series, b: pd.Series, minutes=30, degrees=0.25):
    """`a` kümesindeki her örnek için `b` kümesinde zamanda ve konumda yakın örnekleri bulur.

    Yakınlık (TorNet'in test kümesi için kullandığı kural [V25 §3a]):
      * zaman: son karelerin (end_time) arası en fazla `minutes` dakika,
      * konum: enlem farkı ve boylam farkı ayrı ayrı en fazla `degrees` derece.
    `a` ve `b` dosya adı (filename) dizileridir. Dönen tablo: her satır bir (a, b) komşu çifti.
    """
    cat = catalog.set_index("filename")[["end_time", "lat", "lon", "episode_id"]]
    A, B = cat.loc[a.to_numpy()], cat.loc[b.to_numpy()].sort_values("end_time")
    t_b = B["end_time"].to_numpy()
    lat_b, lon_b, name_b = B["lat"].to_numpy(), B["lon"].to_numpy(), B.index.to_numpy()
    win = np.timedelta64(minutes, "m")
    lo = np.searchsorted(t_b, A["end_time"].to_numpy() - win, side="left")
    hi = np.searchsorted(t_b, A["end_time"].to_numpy() + win, side="right")

    rows = []
    for name, lat, lon, i, j in zip(A.index, A["lat"].to_numpy(), A["lon"].to_numpy(), lo, hi):
        if i == j:
            continue
        near = (np.abs(lat_b[i:j] - lat) <= degrees) & (np.abs(lon_b[i:j] - lon) <= degrees)
        rows += [(name, other) for other in name_b[i:j][near]]
    return pd.DataFrame(rows, columns=["a", "b"])


def neighbor_report(splits: pd.DataFrame, catalog: pd.DataFrame, minutes=30, degrees=0.25) -> pd.DataFrame:
    """Kümeler arasında zaman-konum komşuluğu: kaç örneğin karşı kümede komşusu var?

    episode_id ile bölme, aynı anda ve aynı yerde olan ama farklı fırtına bölümüne ait örnekleri ayırmaz.
    Bu tablo o örtüşmeyi ölçer.
    """
    cat = _with_times(catalog)
    by = {s: splits.loc[splits["split"] == s, "filename"] for s in ["train", "val", "test"]}
    rows = []
    for a, b in [("val", "train"), ("test", "train"), ("test", "val")]:
        pairs = neighbor_pairs(cat, by[a], by[b], minutes, degrees)
        n_a = pairs["a"].nunique()
        rows.append({
            "kümeler": f"{a} → {b}",
            "örnek": len(by[a]),
            "karşı kümede komşusu olan": n_a,
            "oran_%": round(100 * n_a / max(len(by[a]), 1), 2),
            "komşu olan karşı küme örneği": pairs["b"].nunique(),
        })
    return pd.DataFrame(rows).set_index("kümeler")


def drop_val_neighbors(splits: pd.DataFrame, catalog: pd.DataFrame, threshold_pct=1.0,
                       minutes=30, degrees=0.25) -> tuple[pd.DataFrame, dict]:
    """P5 kuralı: doğrulama örneklerinin `threshold_pct` %'sinden fazlasının eğitimde komşusu varsa,
    bu komşu eğitim örnekleri 'dropped' yapılır. Doğrulama kümesine dokunulmaz.

    Döner: (güncellenmiş bölme, özet sözlüğü)
    """
    cat = _with_times(catalog)
    out = splits.copy()
    val = out.loc[out["split"] == "val", "filename"]
    train = out.loc[out["split"] == "train", "filename"]
    pairs = neighbor_pairs(cat, val, train, minutes, degrees)
    pct = 100 * pairs["a"].nunique() / max(len(val), 1)
    info = {"komşusu olan doğrulama örneği_%": round(pct, 2), "eşik_%": threshold_pct,
            "kural uygulandı": pct > threshold_pct, "çıkarılan eğitim örneği": 0}
    if pct > threshold_pct:
        drop = out["filename"].isin(set(pairs["b"])) & (out["split"] == "train")
        out.loc[drop, "split"] = "dropped"
        info["çıkarılan eğitim örneği"] = int(drop.sum())
        info["çıkarılan eğitim hortumu"] = int(out.loc[drop, "label"].sum())
    return out, info


def _with_times(catalog: pd.DataFrame) -> pd.DataFrame:
    """end_time sütunu metinse tarihe çevirir (catalog.csv parse_dates olmadan okunmuş olabilir)."""
    if not np.issubdtype(catalog["end_time"].dtype, np.datetime64):
        catalog = catalog.assign(end_time=pd.to_datetime(catalog["end_time"]))
    return catalog


def split_summary(splits: pd.DataFrame) -> pd.DataFrame:
    """Her kümede kaç örnek, kaç hortum, hortum oranı ve kaç fırtına olduğu."""
    g = splits.groupby("split")
    out = pd.DataFrame({
        "örnek": g.size(),
        "hortum": g["label"].sum(),
        "hortum_%": (100 * g["label"].mean()).round(1),
        "fırtına (episode)": g["episode_id"].nunique(),
    })
    order = [s for s in ["train", "val", "test", "dropped"] if s in out.index]
    return out.loc[order]


def check_splits(splits: pd.DataFrame, catalog: pd.DataFrame, years) -> pd.DataFrame:
    """Bölmenin doğru olduğunu gösteren kontroller. Her satır: kontrol, sonuç, beklenen, tamam mı?"""
    eps = {s: set(splits.loc[splits["split"] == s, "episode_id"]) for s in ["train", "val", "test"]}
    sel = catalog[catalog["year"].isin(years)]
    official_test = sel[sel["type"] == "test"]
    test = splits[splits["split"] == "test"]
    overlap_eps = set(sel.loc[sel["type"] == "train", "episode_id"]) & set(official_test["episode_id"])
    label_ok = (splits["label"] == (splits["category"] == "TOR").astype(int)).all()
    val_nb = neighbor_report(splits, catalog).loc["val → train", "oran_%"]

    rows = [
        ("Eğitim ile doğrulamada ortak fırtına", len(eps["train"] & eps["val"]), 0),
        ("Eğitim ile testte ortak fırtına", len(eps["train"] & eps["test"]), 0),
        ("Doğrulama ile testte ortak fırtına", len(eps["val"] & eps["test"]), 0),
        ("Eğitim + doğrulamada kalan 'ortak' fırtına", len((eps["train"] | eps["val"]) & overlap_eps), 0),
        ("Toplam örnek (eğitim + doğrulama + çıkarılan + test)", len(splits), len(sel)),
        ("Test örneği", len(test), len(official_test)),
        ("Testteki hortum", int(test["label"].sum()), int(official_test["label"].sum())),
        ("Test kümesi hazır test kümesiyle aynı dosyalar mı?", set(test["filename"]) == set(official_test["filename"]), True),
        ("Etiketler kategoriyle uyumlu mu? (TOR = 1)", bool(label_ok), True),
        ("Doğrulama örneklerinin en fazla %1'inin eğitimde komşusu var mı? (30 dk, 0,25°)", bool(val_nb <= 1.0), True),
    ]
    out = pd.DataFrame(rows, columns=["kontrol", "sonuç", "beklenen"])
    out["tamam"] = np.where(out["sonuç"] == out["beklenen"], "✅", "❌")
    return out.set_index("kontrol")
