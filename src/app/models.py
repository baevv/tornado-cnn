"""Model keşfi, önceden hesaplanmış skorlar (preds.parquet), canlı tahmin ve Grad-CAM.

Bir model klasörü (<models_root>/<ad>/) en az best.pt ve results.json içermelidir.
Mimari önce results.json'daki "arch"/"model_kwargs" alanlarından, yoksa configs/experiments.json'dan,
o da yoksa ağırlıkların kendisinden (katman adları ve boyutları) çıkarılır.
torch yalnız canlı tahmin ve Grad-CAM için gerekir; skorlar preds.parquet'ten okunur.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.helper import paths

PREDS_FILE = "preds.parquet"
# Okabe-Ito (renk körlüğüne dayanıklı), sarı yerine koyu mavi; tema ile aynı sıra
MODEL_COLORS = ["#56B4E9", "#009E73", "#0072B2", "#CC79A7", "#D55E00", "#E69F00", "#000000", "#999999"]


@dataclass
class ModelInfo:
    key: str                     # klasör adı
    label: str                   # kısa ad (arayüz)
    long_label: str
    folder: Path
    results: dict
    threshold: float
    params: int | None
    arch: str
    kwargs: dict
    color: str = "#0072B2"
    history: list = field(default_factory=list)

    @property
    def preds_path(self) -> Path:
        return self.folder / PREDS_FILE

    @property
    def has_preds(self) -> bool:
        return self.preds_path.exists()

    @property
    def preds_stale(self) -> bool:
        """best.pt preds.parquet'ten yeniyse skorlar eski modele aittir."""
        return self.has_preds and (self.folder / "best.pt").stat().st_mtime > self.preds_path.stat().st_mtime


def models_root() -> Path:
    return paths.models_root()


def short_label(name: str) -> str:
    """"TornadoCNN (Global Max + BN)" → "CNN · Global Max + BN"; "ResNet-18 (sıfırdan)" → "ResNet-18 · sıfırdan"."""
    m = re.match(r"^\s*(TornadoCNN|ResNet-18)\s*\((.*)\)\s*$", name)
    if not m:
        return name
    head = "CNN" if m.group(1) == "TornadoCNN" else m.group(1)
    return f"{head} · {m.group(2)}"


def _experiments() -> dict:
    try:
        cfg = paths.load_config("experiments.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {e["name"]: e for e in [*cfg.get("baselines", []), *cfg.get("experiments", [])]}


def discover(root: Path | None = None) -> list[ModelInfo]:
    """models_root altındaki her eğitilmiş modeli bulur (best.pt + results.json)."""
    root = Path(root) if root else models_root()
    if not root.exists():
        return []
    exps = _experiments()
    found = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not ((d / "best.pt").exists() and (d / "results.json").exists()):
            continue
        res = json.loads((d / "results.json").read_text(encoding="utf-8"))
        spec = exps.get(re.sub(r"_s\d+$", "", re.sub(r"_hizli$", "", d.name)), {})   # _s7: ek seed (M3)
        long = res.get("model_name") or spec.get("label") or d.name
        hist = []
        if (d / "history.json").exists():
            try:
                hist = json.loads((d / "history.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        found.append(ModelInfo(
            key=d.name,
            label=res.get("short_label") or spec.get("short") or short_label(long),
            long_label=long,
            folder=d,
            results=res,
            threshold=float(res.get("threshold", 0.5)),
            params=res.get("params"),
            arch=res.get("arch") or spec.get("arch") or "",
            kwargs=dict(res.get("model_kwargs") or spec.get("model_kwargs") or {}),
            history=hist,
        ))
    for i, m in enumerate(found):
        m.color = MODEL_COLORS[i % len(MODEL_COLORS)]
    return found


def load_preds(info: ModelInfo) -> pd.Series:
    """filename → hortum olasılığı."""
    df = pd.read_parquet(info.preds_path, columns=["filename", "prob"])
    return df.set_index("filename")["prob"]


# ---------------------------------------------------------------- torch tarafı

def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _infer_arch(state: dict) -> tuple[str, dict]:
    """Ağırlık adlarından mimariyi ve TornadoCNN için filtre/BN/gizli katman boyutlarını çıkarır."""
    if "conv1.weight" in state and any(k.startswith("layer4.") for k in state):
        return "resnet18", {}
    convs = sorted((int(k.split(".")[1]), v) for k, v in state.items()
                   if k.startswith("features.") and k.endswith(".weight") and v.ndim == 4)
    filters = tuple(int(v.shape[0]) for _, v in convs)
    bn = any(k.startswith("features.") and k.endswith("running_mean") for k in state)
    hidden = int(state["head.1.weight"].shape[0]) if "head.1.weight" in state else 64
    return "tornado_cnn", {"filters": filters, "batch_norm": bn, "hidden": hidden}


def load_model(info: ModelInfo, device="cpu"):
    """best.pt'den modeli kurar ve değerlendirme kipine alır."""
    import torch

    from src.helper.model import build_model

    state = torch.load(info.folder / "best.pt", map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state and not any(k.endswith(".weight") for k in state):
        state = state["model"]
    arch, inferred = _infer_arch(state)
    kwargs = {**inferred, **info.kwargs} if (info.arch or arch) == "tornado_cnn" else dict(info.kwargs)
    if arch == "resnet18":
        kwargs["pretrained"] = False                 # ağırlıklar zaten best.pt'de; ImageNet indirme
        kwargs.pop("filters", None)
    variables = info.results.get("variables") or ["DBZ", "VEL", "WIDTH"]
    dropout = float(info.results.get("config", {}).get("dropout", 0.3))
    net = build_model(info.arch or arch, in_channels=len(variables), dropout=dropout, **kwargs)
    net.load_state_dict(state)
    return net.to(device).eval()


def predict_images(net, q: np.ndarray, variables, batch=128, device="cpu", r0_km=None) -> np.ndarray:
    """uint8 (N, C, H, W) → hortum olasılıkları (N,). r0_km (N,): kare başına rng_min, km (CoordConv modeli için)."""
    import torch

    from src.helper.model import bind_range
    from src.helper.train import decode_batch

    out = []
    with torch.no_grad():
        for i in range(0, len(q), batch):
            xb = torch.from_numpy(np.ascontiguousarray(q[i:i + batch])).to(device)
            r = None if r0_km is None else np.asarray(r0_km, dtype=np.float32)[i:i + batch]
            out.append(torch.sigmoid(bind_range(net, r)(decode_batch(xb, variables)).squeeze(1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def gradcam(net, q: np.ndarray, variables, device="cpu", r0_km=None) -> np.ndarray:
    """Her kare için Grad-CAM haritası, 0..1, girdi boyutunda: (N, H, W). Ayrıntı: src/helper/xai.py."""
    from src.helper.model import bind_range
    from src.helper.xai import gradcam as _gradcam
    return _gradcam(bind_range(net, r0_km), q, variables)
