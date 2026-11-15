"""Dosyaların nerede durduğu: Colab'da ve kendi bilgisayarında.

İki tür depolama var:
  * DATA_DIR    - kalıcı, küçük: catalog.csv ve işlenmiş .npz dosyaları.
                  Colab: Google Drive. Yerel: <repo>/data (ya da $TORNADO_DATA_DIR).
  * SCRATCH_DIR - geçici, büyük: indirilen ham TorNet .tar.gz arşivleri.
                  Colab: sanal makinenin diski (/content, oturum kapanınca silinir).
                  Yerel: <repo>/data/raw (ya da $TORNADO_SCRATCH_DIR).

Mac'te Google Drive uygulaması varsa, Colab'ın yazdığı işlenmiş dosyaları kullanmak için
TORNADO_DATA_DIR'i eşitlenen klasöre yönlendir, örneğin:
~/Library/CloudStorage/GoogleDrive-<hesap>/My Drive/tornado-cnn/data
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # bu dosyanın iki üst klasörü = repo kökü


def in_colab() -> bool:
    """Kod Google Colab'da mı çalışıyor?"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def load_config(name: str = "data.json") -> dict:
    """configs/ klasöründeki ayar dosyasını okur."""
    with open(REPO_ROOT / "configs" / name) as f:
        return json.load(f)


def _resolve(env_var: str, colab_default: str, local_default: Path) -> Path:
    """Klasörü seçer: önce ortam değişkeni, yoksa Colab ya da yerel varsayılan. Klasör yoksa oluşturur."""
    if os.environ.get(env_var):
        path = Path(os.environ[env_var]).expanduser()
    elif in_colab():
        path = Path(colab_default)
    else:
        path = local_default
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Kalıcı veri klasörü (katalog ve işlenmiş veri)."""
    sub = load_config()["drive_subdir"]
    return _resolve("TORNADO_DATA_DIR", f"/content/drive/MyDrive/{sub}/data", REPO_ROOT / "data")


def scratch_dir() -> Path:
    """Geçici klasör (ham arşivler)."""
    return _resolve("TORNADO_SCRATCH_DIR", "/content/tornet_raw", REPO_ROOT / "data" / "raw")


def processed_dir() -> Path:
    """İşlenmiş .npz dosyalarının klasörü."""
    path = data_dir() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path
