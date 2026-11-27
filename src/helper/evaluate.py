"""Test değerlendirmesi (Faz 3): karar ölçütleri, NC / RC / WC görünümleri, bootstrap güven aralıkları,
güvenilirlik, EF'ye ve radara uzaklığa göre başarım, grafikler.

Kararlar: M1 (ölçütler), M2 (eşik doğrulamada F1 ≡ CSI ile seçilir), M3 (bootstrap %95 GA), M4 (grafikler),
S2 (NC / RC / WC), O3 (seed ensemble'ı).

Bütün fonksiyonlar yalnızca numpy / pandas / sklearn / matplotlib kullanır; GPU ya da veri dosyası gerektirmez.
`meta` her zaman splits.csv satırlarıdır (category, ef_number, episode_id, rng_min sütunları).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- görünümler (S2)

VIEWS = {
    "NC": "TOR ↔ WRN + NUL (genel)",
    "RC": "TOR ↔ NUL (kolay negatifler)",
    "WC": "TOR ↔ WRN (zor negatifler)",
}


def view_mask(meta: pd.DataFrame, view: str) -> np.ndarray:
    """Görünüme giren örnekler: NC hepsi, RC yalnız TOR + NUL, WC yalnız TOR + WRN."""
    cat = meta["category"].to_numpy()
    if view == "NC":
        return np.ones(len(cat), dtype=bool)
    if view == "RC":
        return np.isin(cat, ["TOR", "NUL"])
    if view == "WC":
        return np.isin(cat, ["TOR", "WRN"])
    raise ValueError(f"bilinmeyen görünüm: {view}")


# ---------------------------------------------------------------- ölçütler (M1, M2)

def decision_scores(y, p, thr: float) -> dict:
    """Eşikteki karar ölçütleri. POD = yakalama, SR = 1 − FAR = isabet, CSI = TP / (TP + FP + FN).
    Bias = (TP + FP) / (TP + FN): 1'den büyükse model gerçekte olduğundan fazla hortum der."""
    y = np.asarray(y) > 0.5
    f = np.asarray(p) >= thr
    tp, fp, fn = int((f & y).sum()), int((f & ~y).sum()), int((~f & y).sum())
    pod = tp / (tp + fn) if tp + fn else float("nan")
    sr = tp / (tp + fp) if tp + fp else float("nan")
    csi = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    bias = (tp + fp) / (tp + fn) if tp + fn else float("nan")
    return {"POD": pod, "SR": sr, "CSI": csi, "F1": f1, "Bias": bias, "TP": tp, "FP": fp, "FN": fn}


def summarize(y, p, thr: float) -> dict:
    """Eşikten bağımsız (PR-AUC, ROC-AUC) + eşikli (POD, SR, CSI, F1, Bias) ölçütler."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y)
    two = len(np.unique(y)) > 1
    return {"PR-AUC": average_precision_score(y, p) if two else float("nan"),
            "ROC-AUC": roc_auc_score(y, p) if two else float("nan"),
            **decision_scores(y, p, thr)}


def view_table(meta: pd.DataFrame, p, thr: float) -> pd.DataFrame:
    """NC / RC / WC satırları: örnek sayısı, hortum oranı ve ölçütler."""
    y = meta["label"].to_numpy()
    p = np.asarray(p)
    rows = []
    for v in VIEWS:
        m = view_mask(meta, v)
        rows.append({"görünüm": v, "n": int(m.sum()), "hortum oranı": float(y[m].mean()), **summarize(y[m], p[m], thr)})
    return pd.DataFrame(rows).set_index("görünüm")


# ---------------------------------------------------------------- bootstrap (M3)

BOOT_STATS = ("PR-AUC", "ROC-AUC", "CSI", "F1")


def bootstrap_indices(groups, n_boot=1000, seed=0) -> list[np.ndarray]:
    """Fırtına (episode_id) bazlı bootstrap: fırtınalar yerine koyarak seçilir, seçilen fırtınanın bütün kareleri alınır.

    Aynı fırtınanın kareleri birbirine çok benzer; kare bazlı bootstrap aralıkları yapay olarak daraltır.
    Aynı `groups` ve `seed` ile her model için aynı örneklemler çıkar; böylece modeller arası fark (Δ) da
    eşleştirilmiş olarak hesaplanabilir.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uniq)))]
    members = [order[starts[i]:starts[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]


def bootstrap(y, p, thr: float, idx: list[np.ndarray], stats=BOOT_STATS) -> pd.DataFrame:
    """Her bootstrap örneklemi için ölçütler (satır = örneklem). Tek sınıflı örneklem NaN verir."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    out = {s: np.full(len(idx), np.nan) for s in stats}
    for b, ix in enumerate(idx):
        yb, pb = y[ix], p[ix]
        if yb.min() == yb.max():
            continue
        if "PR-AUC" in out:
            out["PR-AUC"][b] = average_precision_score(yb, pb)
        if "ROC-AUC" in out:
            out["ROC-AUC"][b] = roc_auc_score(yb, pb)
        if "CSI" in out or "F1" in out:
            d = decision_scores(yb, pb, thr)
            for s in ("CSI", "F1"):
                if s in out:
                    out[s][b] = d[s]
    return pd.DataFrame(out)


def ci(samples, level=0.95) -> tuple[float, float]:
    """Yüzdelik güven aralığı."""
    a = (1 - level) / 2
    lo, hi = np.nanquantile(np.asarray(samples, dtype=float), [a, 1 - a])
    return float(lo), float(hi)


def fmt_ci(value: float, samples, digits=3) -> str:
    lo, hi = ci(samples)
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ---------------------------------------------------------------- güvenilirlik (M4-3)

def reliability(y, p, bins=10) -> pd.DataFrame:
    """Eşit genişlikli skor aralıklarında ortalama skor ve gerçekleşen hortum oranı."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for i in range(bins):
        m = b == i
        rows.append({"alt": edges[i], "üst": edges[i + 1], "n": int(m.sum()),
                     "ortalama skor": float(p[m].mean()) if m.any() else np.nan,
                     "gerçekleşen oran": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


# ---------------------------------------------------------------- EF ve uzaklık (M4-6, M4-7)

EF_ORDER = ["EF0", "EF1", "EF2", "EF3+", "WRN", "NUL"]


def ef_class(meta: pd.DataFrame) -> np.ndarray:
    """TOR → EF0, EF1, EF2, EF3+ (EF bilinmiyorsa "EF?"); diğerleri kategori adı."""
    cat = meta["category"].to_numpy()
    ef = pd.to_numeric(meta["ef_number"], errors="coerce").to_numpy()

    def one(c, e):
        if c != "TOR":
            return c
        if not e >= 0:                  # NaN ya da −1
            return "EF?"
        return "EF3+" if e >= 3 else f"EF{int(e)}"

    return np.array([one(c, e) for c, e in zip(cat, ef)], dtype=object)


def ef_table(meta: pd.DataFrame, p, thr: float) -> pd.DataFrame:
    """Her EF sınıfı ve WRN / NUL için: n, medyan skor, eşiği geçen oran (TOR'da POD, diğerlerinde yanlış alarm)."""
    cls, p = ef_class(meta), np.asarray(p)
    rows = []
    for c in EF_ORDER + (["EF?"] if (cls == "EF?").any() else []):
        m = cls == c
        if m.any():
            rows.append({"sınıf": c, "n": int(m.sum()), "medyan skor": float(np.median(p[m])),
                         "eşiği geçen oran": float((p[m] >= thr).mean())})
    return pd.DataFrame(rows).set_index("sınıf")


WINDOW_HALF_KM = 30.0      # 240 mesafe hücresi × 0,25 km = 60 km'lik pencere; hortum yaklaşık pencerenin ortasında
RANGE_EDGES_KM = (0, 50, 100, 150, 400)


def center_range_km(meta: pd.DataFrame) -> np.ndarray:
    """Pencere merkezinin radara uzaklığı (km): rng_min (metre) + 30 km."""
    return meta["rng_min"].to_numpy(dtype=float) / 1000.0 + WINDOW_HALF_KM


def range_table(meta: pd.DataFrame, p, thr: float, edges=RANGE_EDGES_KM) -> pd.DataFrame:
    """Uzaklık aralıklarına göre başarım. ROC-AUC hortum oranından bağımsız olduğu için aralıklar arası kıyasta
    esas alınır; PR-AUC aralığın hortum oranına bağlıdır (o yüzden tabloda oran da verilir)."""
    r, y, p = center_range_km(meta), meta["label"].to_numpy(), np.asarray(p)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        lab = f"{lo}–{hi} km" if hi < edges[-1] else f"{lo}+ km"
        if not m.any():
            continue
        s = summarize(y[m], p[m], thr)
        rows.append({"uzaklık": lab, "n": int(m.sum()), "hortum": int(y[m].sum()), "hortum oranı": float(y[m].mean()),
                     "ROC-AUC": s["ROC-AUC"], "PR-AUC": s["PR-AUC"], "POD": s["POD"], "CSI": s["CSI"]})
    return pd.DataFrame(rows).set_index("uzaklık")


# ---------------------------------------------------------------- ensemble (O3)

def ensemble(probs: list[np.ndarray]) -> np.ndarray:
    """Seed'lerin skor ortalaması."""
    return np.mean(np.stack([np.asarray(p, dtype=np.float64) for p in probs]), axis=0)


# ---------------------------------------------------------------- grafikler (matplotlib)

def performance_diagram(ax, curves: list, points: list | None = None):
    """Roebber (2009) performans diyagramı: x = SR (isabet), y = POD (yakalama).
    Arka planda CSI eş-eğrileri (dolgu) ve frekans yanlılığı (kesikli doğrular).

    curves: [(etiket, y, p)]; points: [(etiket, y, p, eşik)] seçilen eşikteki karar noktası.
    """
    from sklearn.metrics import precision_recall_curve
    sr = np.linspace(0.001, 1, 300)
    pod = np.linspace(0.001, 1, 300)
    S, P = np.meshgrid(sr, pod)
    csi = 1 / (1 / S + 1 / P - 1)
    cf = ax.contourf(S, P, csi, levels=np.arange(0, 1.01, 0.1), cmap="Greys", alpha=0.35)
    ax.figure.colorbar(cf, ax=ax, label="CSI", pad=0.02)
    for b in (0.5, 1, 1.5, 2, 4):
        x = np.linspace(0, 1, 50)
        ax.plot(x, b * x, color="0.45", lw=0.8, ls="--")
        xt = min(0.93, 0.93 / b)
        ax.text(xt, b * xt, f"{b:g}", fontsize=8, color="0.35", ha="center", va="bottom")
    colors = {}
    for lab, y, p in curves:
        prec, rec, _ = precision_recall_curve(y, p)
        line, = ax.plot(prec[:-1], rec[:-1], lw=1.8, label=lab)      # son nokta (recall 0, precision 1) yapay
        colors[lab] = line.get_color()
    for lab, y, p, thr in points or []:
        d = decision_scores(y, p, thr)
        ax.plot(d["SR"], d["POD"], "o", ms=8, mec="k", color=colors.get(lab, "k"))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Başarı oranı SR = 1 − FAR", ylabel="Yakalama POD", aspect="equal")
    ax.legend(fontsize=8, loc="lower left")
    return ax


def reliability_diagram(ax, ax_hist, items: list, bins=10):
    """items: [(etiket, y, p)]. Üstte güvenilirlik eğrisi (köşegen = kalibre), altta skor histogramı (log)."""
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1)
    edges = np.linspace(0, 1, bins + 1)
    for lab, y, p in items:
        r = reliability(y, p, bins)
        ok = r["n"] > 0
        line, = ax.plot(r.loc[ok, "ortalama skor"], r.loc[ok, "gerçekleşen oran"], "o-", ms=4,
                        label=f"{lab} · Brier {brier(y, p):.3f}")
        ax_hist.hist(p, bins=edges, histtype="step", lw=1.5, color=line.get_color())
    ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Gerçekleşen hortum oranı", title="Güvenilirlik diyagramı")
    ax.legend(fontsize=8, loc="upper left")
    ax_hist.set(xlim=(0, 1), yscale="log", xlabel="Model skoru", ylabel="Kare sayısı")
    return ax


def ef_plot(ax, meta: pd.DataFrame, p, thr: float, title=""):
    """EF0, EF1, EF2, EF3+, WRN, NUL için skor dağılımı (kutu grafiği) ve eşik çizgisi."""
    cls, p = ef_class(meta), np.asarray(p)
    order = [c for c in EF_ORDER if (cls == c).any()]
    data = [p[cls == c] for c in order]
    ax.boxplot(data, showfliers=False, widths=0.6)
    ax.set_xticks(range(1, len(order) + 1), [f"{c}\n(n={len(d)})" for c, d in zip(order, data)])
    ax.axhline(thr, color="crimson", ls="--", lw=1.2, label=f"eşik {thr:.2f}")
    ax.set(ylim=(0, 1), ylabel="Model skoru", title=title or "EF'ye göre skor dağılımı")
    ax.legend(fontsize=8)
    return ax


def range_plot(axes, tables: dict):
    """tables: {etiket: range_table(...)}; sol ROC-AUC, sağ POD (eşikte)."""
    for lab, t in tables.items():
        x = np.arange(len(t))
        axes[0].plot(x, t["ROC-AUC"], "o-", label=lab)
        axes[1].plot(x, t["POD"], "o-", label=lab)
    first = next(iter(tables.values()))
    names = [f"{i}\n({h} hortum)" for i, h in zip(first.index, first["hortum"])]
    for ax, ttl in zip(axes, ["ROC-AUC", "POD (seçilen eşikte)"]):
        ax.set_xticks(range(len(first)), names, fontsize=8)
        ax.set(title=f"Radara uzaklığa göre {ttl}", xlabel="Pencere merkezinin radara uzaklığı")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    return axes
