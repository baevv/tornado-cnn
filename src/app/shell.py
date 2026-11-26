"""Üst bar + Fırtına Gezgini bileşeni (src/app/frontend, derleme gerektirmeyen çift yönlü Streamlit bileşeni).

Python durumu tek bir sözlükle gönderir; bileşen kullanıcı eylemlerini {"n": sıra, "a": eylem, ...} olarak döndürür.
"""
from functools import lru_cache
from pathlib import Path

import streamlit.components.v1 as components

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
_component = components.declare_component("firtina_shell", path=str(HERE / "frontend"))
KEY = "shell"


@lru_cache(maxsize=1)
def _assets(mtimes):
    icons = (ASSETS / "icons.svg").read_text(encoding="utf-8")
    logo_path = ASSETS / "logo.svg"
    logo = logo_path.read_text(encoding="utf-8") if logo_path.exists() else ""
    custom = bool(logo) and "Yer tutucu logo" not in logo
    return icons, logo, custom


def render(state: dict):
    """Bileşeni çizer. Dönen değer: son kullanıcı eylemi (ya da None)."""
    mt = tuple((ASSETS / f).stat().st_mtime if (ASSETS / f).exists() else 0 for f in ("icons.svg", "logo.svg"))
    icons, logo, custom = _assets(mt)
    state = {**state, "icons": icons, "icons_key": str(mt), "logo": logo, "logo_custom": custom}
    return _component(state=state, key=KEY, default=None)
