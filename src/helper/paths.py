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

REPO_ROOT = Path(__file__).resolve().parents[2]   # bu dosyanın üç üst klasörü = repo kökü


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


def models_root() -> Path:
    """Eğitilen modellerin kök klasörü ($TORNADO_MODELS_DIR varsa onu, yoksa data_dir()/../models kullanır)."""
    default = data_dir().parent / "models"
    return _resolve("TORNADO_MODELS_DIR", str(default), default)


def models_dir(name: str) -> Path:
    """Eğitilen bir modelin kalıcı klasörü: <models_root>/<name> (Colab'da Drive)."""
    path = models_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_writable_dir(path: Path) -> bool:
    """Klasörün oluşturulup yazılabildiğini test eder (paylaşılan Drive salt-okunur olabilir)."""
    probe = path / ".write_probe"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def prepare_colab_dirs() -> tuple[Path, Path]:
    """Colab'da veri ve model klasörlerini hazırlar; Drive kısayollarını otomatik bulur.

    Paylaşılan Drive klasörü 'Drive'ım' içine kısayol olarak eklendiyse (adı ne olursa olsun)
    splits.csv'yi bulup TORNADO_DATA_DIR olarak ayarlar. Paylaşılan klasör salt-okunursa
    modellerin yazılacağı yeri kullanıcının kendi Drive'ına (TORNADO_MODELS_DIR) yönlendirir.
    """
    if not in_colab() or os.environ.get("TORNADO_DATA_DIR"):
        return data_dir(), models_root()

    sub = load_config()["drive_subdir"]
    mydrive = Path("/content/drive/MyDrive")
    default_data = mydrive / sub / "data"

    if (default_data / "splits.csv").exists():
        found_data = default_data
    else:
        hits = [*mydrive.glob("*/splits.csv"), *mydrive.glob("*/*/splits.csv"), *mydrive.glob("*/*/*/splits.csv")]
        with_proc = [p.parent for p in hits if (p.parent / "processed").is_dir()]
        candidates = with_proc or [p.parent for p in hits]
        if not candidates:
            raise FileNotFoundError(
                "Drive'ım içinde splits.csv bulunamadı.\n"
                "Paylaşılan Drive klasörünü açıp 'Kısayol ekle → Drive'ım (MyDrive)' yaptıktan sonra "
                "bu hücreyi tekrar çalıştırın."
            )
        found_data = candidates[0]
        os.environ["TORNADO_DATA_DIR"] = str(found_data)

    shared_models = found_data.parent / "models"
    if not os.environ.get("TORNADO_MODELS_DIR") and not _is_writable_dir(shared_models):
        os.environ["TORNADO_MODELS_DIR"] = str(mydrive / sub / "models")

    return data_dir(), models_root()
