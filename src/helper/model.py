"""Hortum tespiti için küçük bir evrişimli sinir ağı (CNN).

Mimari (girdi 3 × 120 × 240):
  4 × [Conv 3×3 → ReLU → MaxPool 2×2]   filtre sayıları 16, 32, 64, 128
  → Global max havuzlama               her özellik haritasından en büyük değer (128 sayı)
  → Dense 64 → ReLU → Dropout
  → Dense 1                            logit; sigmoid kayıp fonksiyonunun içinde uygulanır

Sadece derste görülen katmanlar kullanıldı (evrişim, ReLU, havuzlama, tam bağlı, Dropout).
"""
import torch
from torch import nn


class TornadoCNN(nn.Module):
    """Radar görüntüsünden (kanal × 120 × 240) hortum logit'i üreten CNN."""

    def __init__(self, in_channels=3, filters=(16, 32, 64, 128), hidden=64, dropout=0.3,
                 pool="max", batch_norm=False):
        super().__init__()
        blocks, c_in = [], in_channels
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

    def forward(self, x):
        """x: (B, C, H, W) → (B, 1) logit. Olasılık için torch.sigmoid(logit)."""
        return self.head(self.pool(self.features(x)))


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
    """Her katmanın çıktı boyutu ve parametre sayısı (PyTorch'tan) + 6. haftadaki formülle elle hesap.

    Formül: evrişim = filtre_yüksekliği × filtre_genişliği × giriş_kanalı × filtre_sayısı + filtre_sayısı (bias)
            tam bağlı = giriş × çıkış + çıkış (bias)
    """
    import pandas as pd

    rows, hooks = [], []

    def hook(module, inp, out):
        tracked = (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveMaxPool2d,
                   nn.AdaptiveAvgPool2d, nn.BatchNorm2d, nn.ReLU, nn.Dropout, nn.Flatten)
        if isinstance(module, tracked):
            if isinstance(module, nn.Conv2d):
                kh, kw = module.kernel_size
                b = module.out_channels if module.bias is not None else 0
                manual = kh * kw * module.in_channels * module.out_channels + b
                formula = f"{kh}·{kw}·{module.in_channels}·{module.out_channels}" + (f" + {b}" if b else "")
            elif isinstance(module, nn.Linear):
                b = module.out_features if module.bias is not None else 0
                manual = module.in_features * module.out_features + b
                formula = f"{module.in_features}·{module.out_features}" + (f" + {b}" if b else "")
            elif isinstance(module, nn.BatchNorm2d):
                manual = 2 * module.num_features
                formula = f"2·{module.num_features}"
            else:
                manual, formula = 0, ""
            rows.append({
                "katman": type(module).__name__,
                "çıktı boyutu": " × ".join(str(d) for d in out.shape[1:]),
                "parametre (PyTorch)": sum(p.numel() for p in module.parameters()),
                "parametre (elle)": manual,
                "formül": formula,
            })

    for m in model.modules():
        if m is not model and not list(m.children()):
            hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return pd.DataFrame(rows)
