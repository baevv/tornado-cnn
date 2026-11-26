"""Veriyi eğitim / doğrulama / test kümelerine bölme ve bölmenin kontrolleri.

Kurallar:
  * Test kümesi: veri setinin hazır `test` kısmı. Hiç değiştirilmez, en sona kadar kullanılmaz.
  * Test fırtınalarıyla ortak olan fırtınalar (episode_id) eğitimden çıkarılır (sızıntı).
  * Doğrulama kümesi: kalan `train` kısmından, FIRTINA OLAYINA GÖRE ayrılır. Aynı fırtınanın
    bütün örnekleri ya eğitimde ya doğrulamada olur.
  * Doğrulamada hortum oranı eğitimdekine yakın olsun diye bölme etikete göre dengelenir.
"""
import numpy as np
import pandas as pd

SPLIT_COLUMNS = ["filename", "year", "type", "category", "label", "ef_number", "episode_id", "split"]


def make_splits(catalog: pd.DataFrame, years, val_fraction=0.15, seed=42) -> pd.DataFrame:
    """Seçilen yılların her örneğine bir küme atar: 'train', 'val', 'test' ya da 'dropped'.

    'dropped' = test kümesindeki bir fırtınaya ait olduğu için eğitimden çıkarılan örnek.
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

    return df[SPLIT_COLUMNS].reset_index(drop=True)


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
    ]
    out = pd.DataFrame(rows, columns=["kontrol", "sonuç", "beklenen"])
    out["tamam"] = np.where(out["sonuç"] == out["beklenen"], "✅", "❌")
    return out.set_index("kontrol")
