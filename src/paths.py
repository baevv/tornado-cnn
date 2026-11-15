"""Where things live, on Colab and on a local machine.

Two kinds of storage:
  * DATA_DIR    - persistent, small: catalog.csv and the processed .npz arrays.
                  Colab: Google Drive. Local: <repo>/data (or $TORNADO_DATA_DIR).
  * SCRATCH_DIR - temporary, large: raw TorNet .tar.gz downloads.
                  Colab: the VM disk (/content, wiped when the session ends).
                  Local: <repo>/data/raw (or $TORNADO_SCRATCH_DIR).

On a Mac with the Google Drive app, point TORNADO_DATA_DIR at the synced
folder, e.g. ~/Library/CloudStorage/GoogleDrive-<account>/My Drive/tornado-cnn/data,
to use the same processed files that Colab wrote.
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def load_config(name: str = "data.json") -> dict:
    with open(REPO_ROOT / "configs" / name) as f:
        return json.load(f)


def _resolve(env_var: str, colab_default: str, local_default: Path) -> Path:
    if os.environ.get(env_var):
        path = Path(os.environ[env_var]).expanduser()
    elif in_colab():
        path = Path(colab_default)
    else:
        path = local_default
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    sub = load_config()["drive_subdir"]
    return _resolve("TORNADO_DATA_DIR", f"/content/drive/MyDrive/{sub}/data", REPO_ROOT / "data")


def scratch_dir() -> Path:
    return _resolve("TORNADO_SCRATCH_DIR", "/content/tornet_raw", REPO_ROOT / "data" / "raw")


def processed_dir() -> Path:
    path = data_dir() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path
