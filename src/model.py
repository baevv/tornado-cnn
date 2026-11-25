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

    def __init__(self, in_channels=3, filters=(16, 32, 64, 128), hidden=64, dropout=0.3):
        super().__init__()
        blocks, c_in = [], in_channels
        for c_out in filters:
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),   # padding=1: boyut korunur
                nn.ReLU(),
                nn.MaxPool2d(2),                                     # boyut yarıya iner
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveMaxPool2d(1)     # global max havuzlama: hortum nerede olursa olsun
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


def count_params(model) -> int:
    """PyTorch'un saydığı öğrenilebilir parametre sayısı."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_table(model, input_shape=(3, 120, 240)):
    """Her katmanın çıktı boyutu ve parametre sayısı (PyTorch'tan) + formülle elle hesap.

    Formül: evrişim = filtre_yüksekliği × filtre_genişliği × giriş_kanalı × filtre_sayısı + filtre_sayısı (bias)
            tam bağlı = giriş × çıkış + çıkış (bias)
    """
    import pandas as pd

    rows, hooks = [], []

    def hook(module, inp, out):
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveMaxPool2d, nn.ReLU, nn.Dropout, nn.Flatten)):
            if isinstance(module, nn.Conv2d):
                kh, kw = module.kernel_size
                manual = kh * kw * module.in_channels * module.out_channels + module.out_channels
                formula = f"{kh}·{kw}·{module.in_channels}·{module.out_channels} + {module.out_channels}"
            elif isinstance(module, nn.Linear):
                manual = module.in_features * module.out_features + module.out_features
                formula = f"{module.in_features}·{module.out_features} + {module.out_features}"
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
