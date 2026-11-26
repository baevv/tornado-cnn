"""Eğitim altyapısı: veriyi yükleme, DataLoader, kayıp ağırlıkları, eğitim döngüsü, ölçütler.

03, 04 ve 05 defterleri aynı kodu kullanır; böylece karşılaştırmalar adil olur.
Ayarlar configs/train.json dosyasından gelir.
"""
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Boş piksellerin (uint8'de 0) dolgu değeri, [-1, 1] ölçeğinde (bkz. tornet.from_uint8)
FILL_VALUES = {"DBZ": -1.0, "VEL": 0.0, "WIDTH": -1.0}


# ---------------------------------------------------------------- genel

def set_seed(seed: int):
    """Tekrarlanabilirlik: aynı seed ile aynı başlangıç ağırlıkları ve aynı karıştırma."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """GPU varsa GPU, yoksa CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- veri

def load_splits(data_dir) -> pd.DataFrame:
    """02'de kaydedilen bölmeyi okur (hangi dosya hangi kümede). Bölme yeniden hesaplanmaz."""
    return pd.read_csv(Path(data_dir) / "splits.csv")


def load_data(processed_dir, splits: pd.DataFrame, years, which=("train", "val"), local_dir=None):
    """İşlenmiş .npz dosyalarından istenen kümeleri belleğe alır.

    * local_dir verilirse dosyalar önce oraya kopyalanır (Colab'da Drive yerine hızlı yerel disk).
    * Veri belleğe uint8 olarak alınır; ondalıklı sayıya çevirme her grup için ayrı yapılır.
    * Bellek tasarrufu için diziler baştan doğru boyutta ayrılır ve yıl yıl doldurulur.

    Döner: {küme: {"X": uint8 (N, C, H, W), "y": (N,), "meta": DataFrame}}, kanal adları
    """
    processed_dir = Path(processed_dir)
    sel = splits[splits["year"].isin(years) & splits["split"].isin(which)]
    metas = {s: sel[sel["split"] == s].reset_index(drop=True) for s in which}
    pos = {s: pd.Series(np.arange(len(m)), index=m["filename"]) for s, m in metas.items()}
    out, variables = {}, None
    filled = {s: 0 for s in which}

    for year in years:
        src = processed_dir / f"tornet_{year}.npz"
        if local_dir is not None:
            local = Path(local_dir) / src.name
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists() or local.stat().st_size != src.stat().st_size:
                shutil.copy(src, local)
            src = local
        d = np.load(src)
        X_year, y_year, names = d["X"], d["y"], d["filename"]
        if variables is None:
            variables = [str(v) for v in d["variables"]]
            for s in which:
                out[s] = {"X": np.empty((len(metas[s]),) + X_year.shape[1:], dtype=np.uint8),
                          "y": metas[s]["label"].to_numpy().astype(np.float32),
                          "meta": metas[s]}
        name_s = pd.Series(names)
        for s in which:
            hit = name_s.isin(pos[s].index).to_numpy()
            idx = pos[s].loc[names[hit]].to_numpy()
            out[s]["X"][idx] = X_year[hit]
            filled[s] += len(idx)
            if not np.array_equal(y_year[hit].astype(np.float32), out[s]["y"][idx]):
                raise ValueError(f"{year}: etiketler splits.csv ile uyuşmuyor")
        del X_year, d

    for s in which:
        if filled[s] != len(metas[s]):
            raise ValueError(f"{s}: {len(metas[s])} örnek bekleniyordu, işlenmiş dosyalarda {filled[s]} bulundu")
    return out, variables


class RadarDataset(Dataset):
    """Bellekteki uint8 dizilerden (görüntü, etiket, örnek ağırlığı) döndürür."""

    def __init__(self, X, y, w=None):
        self.X, self.y = X, y
        self.w = np.ones(len(y), dtype=np.float32) if w is None else w.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), self.y[i], self.w[i]


def decode_batch(xb, variables):
    """uint8 grup (B, C, H, W) → [-1, 1] aralığında float; boş pikseller (0) fiziksel değerle doldurulur.

    GPU'da yapılır: veri bellekte 1 bayt kalır, sadece o anki grup 4 bayta çevrilir.
    """
    x = (xb.float() - 1.0) / 254.0 * 2.0 - 1.0
    fill = torch.tensor([FILL_VALUES.get(v, -1.0) for v in variables], device=xb.device).view(1, -1, 1, 1)
    return torch.where(xb == 0, fill.expand_as(x), x)


# ---------------------------------------------------------------- kayıp ağırlıkları

def class_weights(meta: pd.DataFrame, scheme: str):
    """Örnek ağırlıkları ve pos_weight.

    * "pos_weight": her hortum, hortumsuzlara oranla ağırlıklandırılır (hortumsuz / hortum sayısı).
    * "mit":        kategoriye göre ağırlık: rastgele fırtına 1, uyarı (WRN) 0,5, EF0–EF1 1, EF2+ 2.
    * "both":       ikisi birlikte.
    Döner: (örnek ağırlıkları dizisi, pos_weight sayısı)
    """
    y = (meta["category"] == "TOR").to_numpy()
    pos_w = float((~y).sum() / max(y.sum(), 1))
    w = np.ones(len(meta), dtype=np.float32)
    if scheme in ("mit", "both"):
        w[meta["category"].to_numpy() == "WRN"] = 0.5
        w[y & (meta["ef_number"].to_numpy() >= 2)] = 2.0
    if scheme == "mit":
        pos_w = 1.0
    elif scheme not in ("pos_weight", "both"):
        raise ValueError(f"bilinmeyen ağırlık türü: {scheme}")
    return w, pos_w


def make_loaders(data, cfg, weights_train):
    """Eğitim (karıştırmalı) ve doğrulama (karıştırmasız) DataLoader'ları."""
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    pin = torch.cuda.is_available()
    train = DataLoader(RadarDataset(data["train"]["X"], data["train"]["y"], weights_train),
                       batch_size=cfg["batch_size"], shuffle=True, generator=g,
                       num_workers=cfg["num_workers"], pin_memory=pin, drop_last=False)
    val = DataLoader(RadarDataset(data["val"]["X"], data["val"]["y"]),
                     batch_size=cfg["batch_size"] * 2, shuffle=False,
                     num_workers=cfg["num_workers"], pin_memory=pin)
    return train, val


def make_optimizer(model, cfg):
    """Optimizer: adam, sgd, momentum (SGD + momentum 0,9) ya da rmsprop. weight_decay = L2 cezası."""
    name, lr, wd = cfg["optimizer"].lower(), cfg["lr"], cfg.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    if name == "momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"bilinmeyen optimizer: {name}")


# ---------------------------------------------------------------- eğitim

def _loss_fn(pos_weight, device):
    """BCEWithLogitsLoss = sigmoid + ikili çapraz entropi (tek işlemde, sayısal olarak kararlı)."""
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")


@torch.no_grad()
def predict(model, loader, variables, device, pos_weight=1.0):
    """Bir kümenin hortum olasılıkları ve ortalama kaybı."""
    model.eval()
    crit = _loss_fn(pos_weight, device)
    probs, total, n = [], 0.0, 0
    for xb, yb, wb in loader:
        x = decode_batch(xb.to(device, non_blocking=True), variables)
        yb, wb = yb.to(device), wb.to(device)
        logit = model(x).squeeze(1)
        total += (crit(logit, yb) * wb).sum().item()
        n += len(yb)
        probs.append(torch.sigmoid(logit).cpu())
    return torch.cat(probs).numpy(), total / max(n, 1)


def make_scheduler(opt, cfg):
    """Öğrenme oranı planlaması. "plateau": doğrulama AUC'si 2 tur iyileşmezse lr yarıya iner."""
    if cfg.get("scheduler") == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    if cfg.get("scheduler") is None:
        return None
    raise ValueError(f"bilinmeyen planlama: {cfg['scheduler']}")


def _save_checkpoint(path, model, opt, sched, state, cfg):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict() if sched is not None else None,
                "state": state, "cfg": cfg}, path)


def fit(model, train_loader, val_loader, cfg, variables, out_dir, pos_weight, device, log=print):
    """Eğitim döngüsü: her grupta ileri yayılım → kayıp → geri yayılım → güncelleme.

    * Her turdan sonra doğrulama kaybı, ROC-AUC ve PR-AUC hesaplanır.
    * Erken durdurma: doğrulama AUC'si `patience` tur üst üste iyileşmezse durur; en iyi ağırlıklara dönülür.
    * Her tur out_dir'e kaydedilir (last.pt); hücre tekrar çalıştırılınca kaldığı turdan devam eder.
    Döner: tur tur geçmiş (DataFrame)
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg)
    crit = _loss_fn(pos_weight, device)
    state = {"epoch": 0, "history": [], "best_auc": -1.0, "best_epoch": 0, "bad_epochs": 0, "done": False}

    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("cfg") == cfg:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            if sched is not None and ck.get("scheduler"):
                sched.load_state_dict(ck["scheduler"])
            state = ck["state"]
            log(f"Kayıt bulundu: {state['epoch']}. turdan devam ediliyor.")
        else:
            log("Klasörde farklı ayarlarla eğitilmiş bir kayıt var; baştan başlanıyor.")

    y_val = val_loader.dataset.y
    while not state["done"] and state["epoch"] < cfg["epochs"]:
        t0 = time.time()
        lr_now = opt.param_groups[0]["lr"]
        model.train()
        total, n = 0.0, 0
        for xb, yb, wb in train_loader:
            x = decode_batch(xb.to(device, non_blocking=True), variables)
            yb, wb = yb.to(device), wb.to(device)
            logit = model(x).squeeze(1)                 # 1. ileri yayılım
            loss = (crit(logit, yb) * wb).mean()        # 2. kayıp
            opt.zero_grad()
            loss.backward()                             # 3. geri yayılım
            opt.step()                                  # 4. ağırlık güncelleme
            total += loss.item() * len(yb)
            n += len(yb)

        p_val, val_loss = predict(model, val_loader, variables, device, pos_weight)
        auc, pr_auc = roc_auc_score(y_val, p_val), average_precision_score(y_val, p_val)
        state["epoch"] += 1
        row = {"tur": state["epoch"], "eğitim kaybı": total / n, "doğrulama kaybı": val_loss,
               "doğrulama AUC": auc, "doğrulama PR-AUC": pr_auc, "süre (sn)": time.time() - t0, "lr": lr_now}
        state["history"].append(row)
        if sched is not None:
            sched.step(auc)

        mark = ""
        if auc > state["best_auc"]:
            state.update(best_auc=auc, best_epoch=state["epoch"], bad_epochs=0)
            torch.save(model.state_dict(), best_path)
            mark = "  ★ en iyi"
        else:
            state["bad_epochs"] += 1
            if state["bad_epochs"] >= cfg["patience"]:
                state["done"] = True
                mark = f"  → erken durdurma (en iyi tur: {state['best_epoch']})"
        log(f"tur {row['tur']:2d} | eğitim kaybı {row['eğitim kaybı']:.4f} | doğrulama kaybı {val_loss:.4f} | "
            f"AUC {auc:.4f} | PR-AUC {pr_auc:.4f} | {row['süre (sn)']:.0f} sn{mark}")
        _save_checkpoint(last_path, model, opt, sched, state, cfg)

    state["done"] = True
    _save_checkpoint(last_path, model, opt, sched, state, cfg)
    model.load_state_dict(torch.load(best_path, map_location=device))
    history = pd.DataFrame(state["history"])
    history.to_json(out_dir / "history.json", orient="records", indent=1, force_ascii=False)
    log(f"En iyi tur: {state['best_epoch']} (doğrulama AUC {state['best_auc']:.4f}). Bu turun ağırlıkları yüklendi.")
    return history


# ---------------------------------------------------------------- ölçütler

def choose_threshold(y, p):
    """Karar eşiği: doğrulama kümesinde F1'i en yüksek yapan olasılık değeri."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def metrics(y, p, threshold):
    """Eşik ile karar verip ölçütleri hesaplar."""
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "doğruluk": accuracy_score(y, pred),
        "recall (yakalama)": recall_score(y, pred, zero_division=0),
        "precision (isabet)": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "PR-AUC": average_precision_score(y, p) if len(set(y)) > 1 else float("nan"),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def breakdown(meta: pd.DataFrame, p, threshold):
    """Seçilen eşikte kategoriye göre oranlar: hortumlarda yakalama (EF'ye göre), hortumsuzlarda yanlış alarm."""
    pred = np.asarray(p) >= threshold
    cat, ef = meta["category"].to_numpy(), meta["ef_number"].to_numpy()
    tor = cat == "TOR"

    def rate(mask):
        return float(pred[mask].mean()) if mask.any() else float("nan")

    return {"EF0–1 yakalama": rate(tor & (ef < 2)), "EF2+ yakalama": rate(tor & (ef >= 2)),
            "WRN yanlış alarm": rate(cat == "WRN"), "NUL yanlış alarm": rate(cat == "NUL")}


def save_results(out_dir, **items):
    """Sonuçları (ayarlar, eşik, ölçütler) JSON olarak kaydeder."""
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with open(Path(out_dir) / "results.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1, default=conv)
