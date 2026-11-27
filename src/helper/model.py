"""Hortum tespiti için küçük bir evrişimli sinir ağı (CNN).

Mimari (girdi 3 × 120 × 240):
  4 × [Conv 3×3 → ReLU → MaxPool 2×2]   filtre sayıları 16, 32, 64, 128
  → Global max havuzlama               her özellik haritasından en büyük değer (128 sayı)
  → Dense 64 → ReLU → Dropout
  → Dense 1                            logit; sigmoid kayıp fonksiyonunun içinde uygulanır

İsteğe bağlı CoordConv (coord=True, karar K4): girdiye radara uzaklık r ve 1/r kanalları eklenir
(3 radar kanalı + 2 = 5 kanal). Kutupsal ızgarada bir pikselin fiziksel boyutu ve ışının yüksekliği
radara uzaklıkla değişir; evrişim bunu kendisi bilemez. Kanallar örneğin ilk mesafe hücresinin uzaklığından
(rng_min) modelin içinde üretilir, bu yüzden açıklama (XAI) haritaları yalnızca radar kanalları üzerindedir.
"""
import torch
from torch import nn

RANGE_GATE_KM = 0.25     # mesafe hücresi: 250 m
MAX_RANGE_KM = 250.0     # r kanalının ölçek üst sınırı (TorNet pencereleri 0–~250 km)
INV_REF_KM = 10.0        # 1/r ölçeği: 10 km / max(r, 10 km). 1 km ile kanal >50 km'de neredeyse sabit kalıyordu


def range_channels(r0_km: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Her örnek için radara uzaklık kanalları: (B,) ilk hücre uzaklığı, km → (B, 2, H, W), [-1, 1].

    r_j  = rng_min + (j + 0,5) · 0,25 km   (j = 0..W-1, mesafe ekseni son boyut)
    r    : negatifler 0'a kırpılır, [0, 250 km] → [-1, 1]
    1/r  : 10 km / max(r, 10 km) ∈ (0, 1] → [-1, 1]   (ör. 30 km → -0,33; 100 km → -0,8)
    Azimut boyunca sabittir (karar P2).
    """
    j = torch.arange(width, device=r0_km.device, dtype=torch.float32)
    r = r0_km.float().view(-1, 1) + (j + 0.5) * RANGE_GATE_KM          # (B, W)
    r_ch = r.clamp(0.0, MAX_RANGE_KM) / MAX_RANGE_KM * 2.0 - 1.0
    inv_ch = (INV_REF_KM / r.clamp(min=INV_REF_KM)) * 2.0 - 1.0
    ch = torch.stack([r_ch, inv_ch], dim=1)                              # (B, 2, W)
    return ch.unsqueeze(2).expand(-1, -1, height, -1)


class TornadoCNN(nn.Module):
    """Radar görüntüsünden (kanal × 120 × 240) hortum logit'i üreten CNN."""

    def __init__(self, in_channels=3, filters=(16, 32, 64, 128), hidden=64, dropout=0.3,
                 pool="max", batch_norm=False, coord=False):
        super().__init__()
        self.coord = bool(coord)                       # True: forward(x, r0_km) ister
        blocks, c_in = [], in_channels + (2 if coord else 0)
        for c_out in filters:
            blocks.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))   # padding=1: boyut korunur
            if batch_norm:
                blocks.append(nn.BatchNorm2d(c_out))
            blocks += [
                nn.ReLU(),
                nn.MaxPool2d(2),                                              # boyut yarıya iner
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)

        if pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)     # global max havuzlama: hortum nerede olursa olsun
        elif pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)     # global ortalama havuzlama (ablasyon için)
        else:
            raise ValueError(f"bilinmeyen havuzlama türü: {pool} ('max' veya 'avg' olmalı)")

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, r0_km=None):
        """x: (B, C, H, W) → (B, 1) logit. Olasılık için torch.sigmoid(logit).

        coord=True ise r0_km (B,) verilmelidir: her örneğin ilk mesafe hücresinin radara uzaklığı, km.
        """
        if self.coord:
            if r0_km is None:
                raise ValueError("CoordConv modeli r0_km (örnek başına rng_min, km) ister")
            r0_km = torch.as_tensor(r0_km, device=x.device).reshape(-1).expand(x.shape[0])
            x = torch.cat([x, range_channels(r0_km, x.shape[2], x.shape[3]).to(x.dtype)], dim=1)
        return self.head(self.pool(self.features(x)))


def needs_range(net) -> bool:
    """Model ileri geçişte radara uzaklık (r0_km) istiyor mu? (CoordConv)"""
    return bool(getattr(net, "coord", False))


class WithRange(nn.Module):
    """CoordConv modelini tek bir karenin uzaklığına bağlar: forward(x) = net(x, r0_km).

    XAI yöntemleri aynı karenin bozulmuş kopyalarıyla çalışır (occlusion, IG, silme eğrisi);
    hepsi net(x) çağırdığı için bu sarmalayıcıyla değişmeden kullanılabilir.
    """

    def __init__(self, net, r0_km):
        super().__init__()
        self.net = net
        self.r0_km = torch.as_tensor(r0_km, dtype=torch.float32).reshape(-1)   # tek değer ya da kare başına

    @property
    def features(self):                     # Grad-CAM katmanı (xai.cam_layer) içteki modelden bulunur
        return self.net.features

    def forward(self, x):
        r0 = self.r0_km.to(x.device)
        if r0.numel() != x.shape[0]:
            r0 = r0[:1].expand(x.shape[0])  # aynı karenin bozulmuş kopyaları (occlusion, IG, silme eğrisi)
        return self.net(x, r0)


def bind_range(net, r0_km):
    """CoordConv modeliyse uzaklığa bağlanmış sarmalayıcı, değilse modelin kendisi."""
    if not needs_range(net):
        return net
    if r0_km is None:
        raise ValueError("CoordConv modeli için kare başına rng_min (km) gerekli")
    return WithRange(net, r0_km)


def make_resnet18(in_channels=3, pretrained=True, dropout=0.3):
    """İkili hortum sınıflandırması için ResNet-18 (transfer öğrenme veya sıfırdan).

    Çıktı: (B, 1) logit (TornadoCNN ile aynı arayüz; train.py ile doğrudan uyumlu).
    """
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    net = resnet18(weights=weights)
    if in_channels != 3:
        old = net.conv1
        net.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                              stride=old.stride, padding=old.padding, bias=False)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1)) if dropout > 0 else nn.Linear(in_features, 1)
    return net


def build_model(arch: str, in_channels=3, dropout=0.3, **kwargs):
    """Config adından model kurar ('tornado_cnn' veya 'resnet18')."""
    key = arch.lower()
    if key == "tornado_cnn":
        return TornadoCNN(in_channels=in_channels, dropout=dropout, **kwargs)
    if key == "resnet18":
        return make_resnet18(in_channels=in_channels, dropout=dropout, **kwargs)
    raise ValueError(f"bilinmeyen mimari: {arch} ('tornado_cnn' veya 'resnet18' olmalı)")


def count_params(model) -> int:
    """PyTorch'un saydığı öğrenilebilir parametre sayısı."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_table(model, input_shape=(3, 120, 240)):
    """Her katmanın çıktı boyutu, eğitilebilir parametre sayısı ve hesaplama formülü.

    Formül: evrişim  = (filtre_yüksekliği · filtre_genişliği · giriş_kanalı + 1) · filtre_sayısı
            tam bağlı = (giriş + 1) · çıkış            (+1: her nöronun bias'ı)
    Formülle bulunan sayı PyTorch'un saydığıyla karşılaştırılır (assert).
    """
    import pandas as pd

    rows, hooks = [], []

    def hook(module, inp, out):
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            has_bias = module.bias is not None
            manual = (kh * kw * module.in_channels + (1 if has_bias else 0)) * module.out_channels
            formula = f"({kh}·{kw}·{module.in_channels}" + (" + 1" if has_bias else "") + f") · {module.out_channels}"
        elif isinstance(module, nn.Linear):
            has_bias = module.bias is not None
            manual = (module.in_features + (1 if has_bias else 0)) * module.out_features
            formula = f"({module.in_features}" + (" + 1" if has_bias else "") + f") · {module.out_features}"
        elif isinstance(module, nn.BatchNorm2d):
            manual = 2 * module.num_features if module.affine else 0
            formula = f"2 · {module.num_features}" if module.affine else ""
        else:
            manual, formula = 0, ""
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        assert n == manual, f"{type(module).__name__}: PyTorch {n} ≠ formül {manual}"
        rows.append({
            "katman": type(module).__name__,
            "çıktı boyutu": " × ".join(str(d) for d in out.shape[1:]),
            "parametre": n,
            "formül": formula,
        })

    for m in model.modules():
        if m is not model and not list(m.children()):
            hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x0 = torch.zeros(1, *input_shape, device=next(model.parameters()).device)
        model(x0, torch.zeros(1, device=x0.device)) if needs_range(model) else model(x0)
    model.train(was_training)
    for h in hooks:
        h.remove()
    table = pd.DataFrame(rows)
    assert table["parametre"].sum() == count_params(model)
    return table
