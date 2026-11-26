"""Açıklanabilir yapay zeka (XAI): model bir radar karesinde neye bakarak karar veriyor?

Dört yöntem birbirini tamamlar:

  Grad-CAM              Son evrişim katmanının çıktısını, skorun o çıktıya göre gradyanıyla ağırlıklandırır.
                        "Model NEREYE bakıyor?" Kaba çözünürlük (TornadoCNN'de 15 × 30, ResNet-18'de 4 × 8).
  Integrated Gradients  Boş girdiden (yankı yok) gerçek kareye giden düz yol boyunca gradyanların ortalaması
                        × girdi farkı. Piksel ve kanal düzeyinde; katkıların toplamı skor farkına eşittir
                        (tamlık özelliği, logit ölçeğinde).
  Occlusion             Görüntünün bir yaması "yankı yok" ile kapatılır, skorun ne kadar düştüğü ölçülür.
                        Modelin içine bakmaz, doğrudan etkiyi ölçer (LIME'ın sorduğu soruyu rastgelelik olmadan sorar).
  Kanal SHAP            DBZ, VEL ve WIDTH kanallarının katkısı: kesin Shapley değeri. 3 kanalda 2³ = 8 alt küme,
                        yani 8 ileri geçiş; yaklaşık değil. Katkıların toplamı = skor − taban skor.

Açıklamanın kendisi de sınanır:
  Silme eğrisi (deletion)   En önemli pikseller önce silinir; skor ne kadar hızlı düşerse açıklama o kadar sadıktır.
                            Eğri altındaki alan rastgele sırayla karşılaştırılır.
  Fiziksel tutarlılık       Açıklamanın zirvesi ile en güçlü dönmenin (VEL'deki hız çifti) arası, km.
  Sağlamlık kontrolü        Ağırlıklar rastgele yapılınca harita değişmeli (Adebayo vd., 2018);
                            eğitilmiş ve rastgele modelin haritaları arasındaki sıra korelasyonu düşük olmalı.

"Silmek" her yerde aynı anlama gelir: pikselin uint8 değeri 0 yapılır, yani yankı yok. decode_batch bu pikseli
eğitimde de kullanılan doldurma değeriyle (FILL_VALUES) doldurur; model için tanıdık bir "boş" değerdir.

Bütün fonksiyonlar uint8 girdi alır: (N, C, H, W), arayüzün ve npy dosyalarının biçimi.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.helper.train import decode_batch

METHODS = {"cam": "Grad-CAM", "ig": "Integrated Gradients", "occ": "Occlusion"}
DAZ, DR = 0.5, 0.25          # TorNet: açı adımı (derece), mesafe adımı (km)


def _device(net) -> torch.device:
    return next(net.parameters()).device


def _x(q: np.ndarray, variables, device) -> torch.Tensor:
    """uint8 (N, C, H, W) → modelin girdisi (float, [-1, 1])."""
    return decode_batch(torch.from_numpy(np.ascontiguousarray(q)).to(device), list(variables))


@torch.no_grad()
def predict(net, q: np.ndarray, variables, batch=128) -> np.ndarray:
    """uint8 (N, C, H, W) → hortum olasılıkları (N,)."""
    dev, out = _device(net), []
    for i in range(0, len(q), batch):
        out.append(torch.sigmoid(net(_x(q[i:i + batch], variables, dev)).squeeze(1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------- Grad-CAM

def cam_layer(net):
    """Grad-CAM katmanı: ResNet'te layer4, TornadoCNN'de son ReLU (son evrişim bloğunun havuzlamadan önceki çıktısı)."""
    if hasattr(net, "layer4"):
        return net.layer4
    relus = [m for m in net.features if isinstance(m, nn.ReLU)]
    return relus[-1]


def gradcam(net, q: np.ndarray, variables) -> np.ndarray:
    """Her kare için Grad-CAM haritası, 0..1, girdi boyutunda: (N, H, W)."""
    layer = cam_layer(net)
    acts, grads = {}, {}

    def keep(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))

    h = layer.register_forward_hook(keep)
    try:
        x = _x(q, variables, _device(net)).requires_grad_(True)
        net.zero_grad()
        net(x).squeeze(1).sum().backward()
        a, g = acts["a"], grads["g"]
        w = g.mean(dim=(2, 3), keepdim=True)                     # her filtrenin önemi
        cam = F.relu((w * a).sum(1, keepdim=True))               # yalnız hortum lehine olan bölgeler
        cam = F.interpolate(cam, size=q.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam.detach().cpu().numpy()
    finally:
        h.remove()
    return _norm_max(cam)


# ---------------------------------------------------------------- Integrated Gradients

def integrated_gradients(net, q: np.ndarray, variables, steps=32) -> np.ndarray:
    """Tek kare (C, H, W) için IG katkıları (C, H, W), logit ölçeğinde, işaretli.

    Taban: bütün pikseller boş (uint8 0). Riemann orta nokta kuralı, `steps` adım.
    Tamlık: katkıların toplamı ≈ logit(kare) − logit(taban).
    """
    dev = _device(net)
    x = _x(q[None], variables, dev)
    x0 = _x(np.zeros_like(q)[None], variables, dev)
    alphas = (torch.arange(steps, device=dev, dtype=x.dtype) + 0.5) / steps
    path = (x0 + alphas.view(-1, 1, 1, 1) * (x - x0)).requires_grad_(True)
    net.zero_grad()
    net(path).squeeze(1).sum().backward()
    avg_grad = path.grad.mean(0, keepdim=True)
    return ((x - x0) * avg_grad)[0].detach().cpu().numpy()


# ---------------------------------------------------------------- Occlusion

@torch.no_grad()
def occlusion(net, q: np.ndarray, variables, patch=10, batch=96) -> np.ndarray:
    """Tek kare (C, H, W): her `patch` × `patch` yama sırayla kapatılınca olasılıktaki düşüş, (H, W).

    Pozitif = o bölge hortum kararını destekliyor. 120 × 240 ve patch=10 için 12 × 24 = 288 ileri geçiş.
    """
    C, H, W = q.shape
    base = float(predict(net, q[None], variables)[0])
    cells = [(i, j) for i in range(0, H, patch) for j in range(0, W, patch)]
    drops = np.zeros(len(cells), dtype=np.float32)
    for s in range(0, len(cells), batch):
        chunk = cells[s:s + batch]
        qb = np.repeat(q[None], len(chunk), 0)
        for k, (i, j) in enumerate(chunk):
            qb[k, :, i:i + patch, j:j + patch] = 0
        drops[s:s + len(chunk)] = base - predict(net, qb, variables, batch=batch)
    out = np.zeros((H, W), dtype=np.float32)
    for (i, j), d in zip(cells, drops):
        out[i:i + patch, j:j + patch] = d
    return out


# ---------------------------------------------------------------- Kanal SHAP

@torch.no_grad()
def channel_shapley(net, q: np.ndarray, variables) -> dict:
    """Kanalların kesin Shapley değerleri, olasılık ölçeğinde.

    q: (N, C, H, W). Bir kanalın "yokluğu" = o kanal tamamen boş (uint8 0).
    Döner: {"base": (N,) hiçbir kanal yokken skor, "score": (N,) gerçek skor, "phi": (N, C)}.
    base + phi.sum(1) == score (Shapley'nin verimlilik özelliği).
    """
    n, C = len(q), q.shape[1]
    subsets = list(itertools.product([0, 1], repeat=C))         # 2^C alt küme
    v = {}
    for s in subsets:
        qs = q.copy()
        for c in range(C):
            if not s[c]:
                qs[:, c] = 0
        v[s] = predict(net, qs, variables)
    phi = np.zeros((n, C), dtype=np.float64)
    for c in range(C):
        for s in subsets:
            if s[c]:
                continue
            k = sum(s)
            w = math.factorial(k) * math.factorial(C - k - 1) / math.factorial(C)
            with_c = tuple(1 if i == c else s[i] for i in range(C))
            phi[:, c] += w * (v[with_c] - v[s])
    return {"base": v[(0,) * C], "score": v[(1,) * C], "phi": phi.astype(np.float32)}


# ---------------------------------------------------------------- Görselleştirme için haritalar

def _norm_max(m: np.ndarray) -> np.ndarray:
    """(N, H, W) ya da (H, W) → her harita kendi en büyük değerine bölünür; 0..1."""
    m = np.asarray(m, dtype=np.float32)
    mx = m.max(axis=(-2, -1), keepdims=True)
    return np.where(mx > 0, m / np.maximum(mx, 1e-12), 0.0).astype(np.float32)


def display_map(attr: np.ndarray, sigma=1.5) -> np.ndarray:
    """İşaretli katkı (H, W) → gösterilecek harita, 0..1: pozitif kısım, hafif yumuşatılmış, %99,5'lik değere göre.

    IG piksel piksel gürültülüdür; yumuşatma yalnız göz için, ölçümlerde ham harita kullanılır.
    """
    from scipy.ndimage import gaussian_filter
    a = gaussian_filter(np.clip(attr, 0, None).astype(np.float32), sigma) if sigma else np.clip(attr, 0, None)
    hi = np.percentile(a, 99.5)
    return np.clip(a / hi, 0, 1).astype(np.float32) if hi > 0 else np.zeros_like(a, dtype=np.float32)


def peak(m: np.ndarray, sigma=2.0) -> tuple[int, int] | None:
    """Haritanın zirvesi (açı, mesafe hücresi); tek piksellik gürültü zirve sayılmasın diye önce yumuşatılır."""
    from scipy.ndimage import gaussian_filter
    s = gaussian_filter(np.asarray(m, dtype=np.float32), sigma)
    if not np.isfinite(s).any() or s.max() <= 0:
        return None
    i, j = np.unravel_index(int(np.argmax(s)), s.shape)
    return int(i), int(j)


def cell_km(a: tuple, b: tuple, az0: float, r0: float) -> float:
    """İki (açı, mesafe) hücresi arasındaki gerçek uzaklık, km (kutupsal → kartezyen)."""
    def xy(c):
        az = math.radians(az0 + (c[0] + 0.5) * DAZ)
        r = r0 + (c[1] + 0.5) * DR
        return r * math.sin(az), r * math.cos(az)
    (x1, y1), (x2, y2) = xy(a), xy(b)
    return float(math.hypot(x1 - x2, y1 - y2))


# ---------------------------------------------------------------- Sadakat: silme eğrisi

@torch.no_grad()
def deletion_curve(net, q: np.ndarray, variables, saliency: np.ndarray, max_frac=0.5, steps=20,
                   rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tek kare (C, H, W): pikseller önem sırasıyla (bütün kanallarda) silinirken skor.

    saliency=None → rastgele sıra (karşılaştırma tabanı). Döner: (silinen oran, skor), her biri (steps + 1,).
    """
    C, H, W = q.shape
    n = H * W
    if saliency is None:
        order = (rng or np.random.default_rng(0)).permutation(n)
    else:
        order = np.argsort(-np.asarray(saliency, dtype=np.float32).ravel(), kind="stable")
    fracs = np.linspace(0, max_frac, steps + 1)
    qb = np.repeat(q[None], len(fracs), 0).reshape(len(fracs), C, n)
    for k, f in enumerate(fracs):
        qb[k][:, order[:int(round(f * n))]] = 0
    return fracs, predict(net, qb.reshape(len(fracs), C, H, W), variables)


_trapz = getattr(np, "trapezoid", None) or np.trapz


def curve_area(fracs: np.ndarray, scores: np.ndarray) -> float:
    """Silme eğrisinin normalleştirilmiş alanı (0..1): x ekseni [0, max_frac] → [0, 1]. Düşük = sadık açıklama."""
    return float(_trapz(scores, fracs) / (fracs[-1] - fracs[0]))


# ---------------------------------------------------------------- Sağlamlık kontrolü

def randomized(net, seed=0):
    """Aynı mimari, bütün öğrenilebilir katmanları yeniden başlatılmış kopya (ağırlık rastgeleleştirme testi)."""
    torch.manual_seed(seed)
    r = copy.deepcopy(net)
    for m in r.modules():
        if hasattr(m, "reset_parameters") and m is not r:
            m.reset_parameters()
        if isinstance(m, nn.BatchNorm2d):                      # eval kipinde kayıtlı istatistikler de sıfırlanır
            m.reset_running_stats()
    return r.eval()


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman sıra korelasyonu (iki haritanın piksel sıralaması ne kadar benziyor)."""
    from scipy.stats import spearmanr
    a, b = np.ravel(a), np.ravel(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


# ---------------------------------------------------------------- Tek kare: hepsi bir arada

def explain_frame(net, q: np.ndarray, variables, rot_cell=None, az0=0.0, r0=0.0, ig_steps=32,
                  patch=10, del_steps=20, n_random=3) -> dict:
    """Bir kare (C, H, W) için bütün açıklamalar ve ölçüler.

    Döner:
      score, base, phi (C,)               kanal SHAP (olasılık)
      maps {cam, ig, occ}: (H, W) 0..1    gösterim haritaları
      ig_ch: (C, H, W) 0..1               IG, kanal kanal (aynı ölçek)
      peaks {yöntem: (i, j)}, dist_km {yöntem: km}   açıklama zirvesi ve en güçlü dönmeye uzaklığı
      deletion {yöntem | "rnd": (oran, skor)}, auc {yöntem | "rnd": alan}
    """
    variables = list(variables)
    cam = gradcam(net, q[None], variables)[0]
    ig = integrated_gradients(net, q, variables, steps=ig_steps)
    occ = occlusion(net, q, variables, patch=patch)
    sh = channel_shapley(net, q[None], variables)

    ig_sum = ig.sum(0)
    raw = {"cam": cam, "ig": np.clip(ig_sum, 0, None), "occ": np.clip(occ, 0, None)}
    maps = {"cam": cam, "ig": display_map(ig_sum), "occ": _norm_max(np.clip(occ, 0, None))}
    ig_pos = np.clip(ig, 0, None)
    from scipy.ndimage import gaussian_filter
    ig_pos = np.stack([gaussian_filter(c, 1.5) for c in ig_pos])
    hi = np.percentile(ig_pos, 99.5)
    ig_ch = np.clip(ig_pos / hi, 0, 1) if hi > 0 else np.zeros_like(ig_pos)

    peaks = {k: peak(maps[k]) for k in maps}
    dist = {k: (cell_km(p, rot_cell, az0, r0) if (p is not None and rot_cell is not None) else None) for k, p in peaks.items()}

    deletion, auc = {}, {}
    for k in METHODS:
        f, s = deletion_curve(net, q, variables, raw[k], steps=del_steps)
        deletion[k], auc[k] = (f, s), curve_area(f, s)
    rng = np.random.default_rng(0)
    rs = [deletion_curve(net, q, variables, None, steps=del_steps, rng=rng) for _ in range(n_random)]
    f = rs[0][0]
    s = np.mean([r[1] for r in rs], 0)
    deletion["rnd"], auc["rnd"] = (f, s), curve_area(f, s)

    return {"score": float(sh["score"][0]), "base": float(sh["base"][0]), "phi": sh["phi"][0],
            "maps": maps, "ig_ch": ig_ch.astype(np.float32), "peaks": peaks, "dist_km": dist,
            "deletion": deletion, "auc": auc}
