"""Fırtına olayı (episode_id) bazlı TorNet .npz + catalog.csv görselleştirici.

Özellikler:
  * ~/tor_npz (veya ~ / data/processed) içindeki tornet_*.npz dosyalarını RAM'de
    kopyalamadan yükler ve catalog.csv ile filename üzerinden eşleştirir.
  * Örnekleri episode_id altında zaman sırasına (start_time) göre gruplar.
  * Üst seviyede fırtına olayları arasında, alt seviyede ise seçili fırtınanın
    tüm zaman kareleri arasında gezinme + Oynat/Durdur + tıklanabilir zaman çizelgesi sunar.
  * Üst satırda CNN'in gördüğü 120×240 ham matrisi, alt satırda ise her değişkenin
    insan gözüyle anlaşılabilen gerçek dünya (kutupsal -> x, y km) görünümünü çizer.
  * Tüm panellerde senkron imleç takibi ve NaN (veri yok) ayrımı sunar.
"""
from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, RadioButtons, TextBox
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RANGES = {"DBZ": (-20.0, 60.0), "VEL": (-60.0, 60.0), "WIDTH": (0.0, 16.0)}
FILL_NORM = {"DBZ": -1.0, "VEL": 0.0, "WIDTH": -1.0}
CMAPS = {"DBZ": "turbo", "VEL": "seismic", "WIDTH": "magma"}
UNITS = {"DBZ": "dBZ", "VEL": "m/s", "WIDTH": "m/s"}

# Tek merkezden yönetilen görsel tema ve düzen ayarları
THEME = {
    "colors": {
        "bg_fig": "#f4f6f8",
        "bg_panel": "#ffffff",
        "bg_panel_alt": "#f8fafc",
        "bg_world": "#f8fafc",
        "bg_btn": "#ffffff",
        "bg_btn_hover": "#eef2f6",
        "bg_play": "#e8f1ff",
        "bg_play_hover": "#d4e6ff",
        "border": "#d0d7de",
        "border_subtle": "#e2e8f0",
        "grid_world": "#cbd5e1",
        "text": "#1f2328",
        "text_muted": "#656d76",
        "accent": "#0969da",
        "safe": "#1a7f37",
        "safe_bg": "#dafbe1",
        "safe_border": "#4ac26b",
        "warn": "#cf222e",
        "warn_bg": "#ffebe9",
        "warn_border": "#ff8182",
        "nan": "#e2e8f0",
        "cursor_outer": "#0f172a",
        "cursor_inner": "#00e5ff",
        "timeline_line": "#cbd5e1",
    },
    "fonts": {
        "sans": ["SF Pro Text", "SF Pro Display", "Inter", "DejaVu Sans", "Arial"],
        "mono": ["SF Mono", "Menlo", "Monaco", "DejaVu Sans Mono"],
    },
    "font_sizes": {
        "badge": 16.0,
        "ep_title": 11.2,
        "filepath": 8.5,
        "panel_title": 12.0,
        "plot_title": 11.5,
        "body": 10.0,
        "mono": 8.8,
        "axis_label": 9.5,
        "tick": 8.5,
        "legend": 8.5,
    },
    "layout": {
        "figsize": (16.0, 9.4),
        "plot_border_lw": 1.6,
        "panel_border_lw": 1.0,
        "grid_world_lw": 0.6,
        "grid_world_alpha": 0.55,
        "cbar_fraction": 0.032,
        "cbar_pad": 0.032,
        "cbar_shrink": 0.72,
        "cbar_aspect": 26,
    },
}


def apply_theme():
    """Global matplotlib yazı tipi ve temel görünüm ayarlarını uygular."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": THEME["fonts"]["sans"],
        "font.monospace": THEME["fonts"]["mono"],
        "text.color": THEME["colors"]["text"],
        "axes.labelcolor": THEME["colors"]["text"],
        "xtick.color": THEME["colors"]["text_muted"],
        "ytick.color": THEME["colors"]["text_muted"],
        "axes.edgecolor": THEME["colors"]["border"],
        "figure.facecolor": THEME["colors"]["bg_fig"],
    })


def find_npz_files() -> list[Path]:
    """Önce ~/tor_npz, yoksa ~, yoksa repo/data/processed içindeki .npz dosyalarını bulur."""
    candidates = [
        Path.home() / "tor_npz",
        Path.home(),
        REPO_ROOT / "data" / "processed",
    ]
    for folder in candidates:
        if folder.exists():
            files = sorted(folder.glob("tornet_*.npz"))
            if files:
                return files
    return []


def find_catalog() -> Path | None:
    """catalog.csv dosyasını ev dizini veya repo içinde arar."""
    candidates = [
        Path.home() / "catalog.csv",
        Path.home() / "tor_npz" / "catalog.csv",
        REPO_ROOT / "data" / "catalog.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_splits() -> dict[str, str]:
    """Varsa splits.csv dosyasından filename -> split eşlemesini okur."""
    candidates = [
        Path.home() / "splits.csv",
        Path.home() / "tor_npz" / "splits.csv",
        REPO_ROOT / "data" / "splits.csv",
    ]
    for path in candidates:
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                return {r["filename"]: r["split"] for r in csv.DictReader(f)}
    return {}


def decode_for_model(q: np.ndarray, variables: list[str]) -> np.ndarray:
    """uint8 (3, 120, 240) -> [-1, 1] float32."""
    x = (q.astype(np.float32) - 1.0) / 254.0 * 2.0 - 1.0
    for i, v in enumerate(variables):
        x[i][q[i] == 0] = FILL_NORM.get(v, -1.0)
    return x


def decode_physical_masked(q: np.ndarray, variables: list[str]) -> list[np.ma.MaskedArray]:
    """uint8 (3, 120, 240) -> fiziksel birimli maskeli diziler (q==0 olanlar NaN maskeli)."""
    out = []
    for i, v in enumerate(variables):
        lo, hi = RANGES[v]
        phys = (q[i].astype(np.float32) - 1.0) / 254.0 * (hi - lo) + lo
        out.append(np.ma.masked_where(q[i] == 0, phys))
    return out


def sector_polygon_xy(az_a: float, az_b: float, r_a: float, r_b: float, n: int = 48) -> np.ndarray:
    """Kutupsal (az_a..az_b derece, r_a..r_b km) diliminin kapalı (x, y) poligon köşeleri."""
    az1 = np.deg2rad(np.linspace(az_a, az_b, n))
    az2 = np.deg2rad(np.linspace(az_b, az_a, n))
    x = np.concatenate([r_a * np.sin(az1), r_b * np.sin(az2)])
    y = np.concatenate([r_a * np.cos(az1), r_b * np.cos(az2)])
    return np.column_stack([x, y])


def physical_stats(q: np.ndarray, variables: list[str]) -> str:
    """Kanal başına fiziksel min/max ve boş piksel yüzdesini satır satır biçimlendirir."""
    lines = []
    for i, v in enumerate(variables):
        lo, hi = RANGES[v]
        ok = q[i] > 0
        empty = 100.0 * float((~ok).mean())
        unit = UNITS.get(v, "")
        if ok.any():
            vals = (q[i][ok].astype(np.float32) - 1.0) / 254.0 * (hi - lo) + lo
            lines.append(f"{v:5s} [{vals.min():5.1f},{vals.max():5.1f}] {unit:3s} · boş %{empty:2.0f}")
        else:
            lines.append(f"{v:5s} tamamen boş")
    return "\n".join(lines)


class EpisodeDataset:
    """Tüm .npz dosyalarını catalog.csv ile birleştirip episode_id altında gruplar."""

    def __init__(self, npz_paths: list[Path], catalog_path: Path | None):
        if not npz_paths:
            raise FileNotFoundError("~/tor_npz veya ~ içinde tornet_*.npz dosyası bulunamadı.")
        if catalog_path is None:
            raise FileNotFoundError("~/catalog.csv dosyası bulunamadı.")

        self.chunks = []
        self.variables: list[str] = []
        file_index: dict[str, tuple[int, int, int]] = {}

        for cid, path in enumerate(npz_paths):
            print(f"Yükleniyor: {path} ...")
            with np.load(path) as d:
                X = d["X"]
                y = d["y"]
                names = d["filename"]
                if not self.variables:
                    self.variables = [str(v) for v in d["variables"]]
            self.chunks.append(X)
            for local_idx, (name, label) in enumerate(zip(names, y)):
                file_index[str(name)] = (cid, local_idx, int(label))

        splits_map = find_splits()
        print(f"Katalog okunuyor: {catalog_path} ...")
        ep_frames: dict[str, list[dict]] = {}
        test_episodes: set[str] = set()

        with open(catalog_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fname = row["filename"]
                ref = file_index.get(fname)
                if ref is None:
                    continue
                cid, local_idx, y_val = ref
                ep_id = row["episode_id"]
                row_type = row["type"]
                if row_type == "test":
                    test_episodes.add(ep_id)

                ef_raw = float(row["ef_number"]) if row["ef_number"] else -1.0
                ef = int(round(ef_raw)) if ef_raw >= 0 else -1
                rng_min = float(row["rng_min"]) / 1000.0
                rng_max = float(row["rng_max"]) / 1000.0
                az_min = float(row["az_min"])
                az_max = float(row["az_max"])

                ep_frames.setdefault(ep_id, []).append({
                    "filename": fname,
                    "chunk_id": cid,
                    "local_idx": local_idx,
                    "y": y_val,
                    "category": row["category"],
                    "ef": ef,
                    "type": row_type,
                    "episode_id": ep_id,
                    "site": row["site_name"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "year": row["start_time"][:4],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "rng_km": (rng_min, rng_max),
                    "az_deg": (az_min, az_max),
                })

        self.episodes: list[dict] = []
        self.ep_id_to_pos: dict[str, int] = {}

        for ep_id, frames in ep_frames.items():
            frames.sort(key=lambda r: (r["start_time"], r["end_time"], r["filename"]))
            if splits_map:
                ep_splits = {splits_map.get(fr["filename"], fr["type"]) for fr in frames}
                split_txt = "/".join(sorted(ep_splits))
            elif ep_id in test_episodes:
                has_train = any(fr["type"] == "train" for fr in frames)
                split_txt = "test + dropped" if has_train else "test"
            else:
                split_txt = "train/val"

            n_tor = sum(1 for fr in frames if fr["y"] == 1)
            n_wrn = sum(1 for fr in frames if fr["category"] == "WRN")
            n_nul = sum(1 for fr in frames if fr["category"] == "NUL")
            max_ef = max(fr["ef"] for fr in frames)

            self.episodes.append({
                "episode_id": ep_id,
                "year": frames[0]["year"],
                "frames": frames,
                "n_frames": len(frames),
                "n_tor": n_tor,
                "n_wrn": n_wrn,
                "n_nul": n_nul,
                "max_ef": max_ef,
                "has_tor": n_tor > 0,
                "has_wrn": n_wrn > 0,
                "split": split_txt,
                "start_time": frames[0]["start_time"],
            })

        self.episodes.sort(key=lambda ep: (ep["start_time"], int(ep["episode_id"])))
        for idx, ep in enumerate(self.episodes):
            self.ep_id_to_pos[ep["episode_id"]] = idx

        total_samples = sum(ep["n_frames"] for ep in self.episodes)
        tor_eps = sum(1 for ep in self.episodes if ep["has_tor"])
        print(
            f"Hazır: {len(self.episodes):,} fırtına olayı ({tor_eps:,} hortumlu), "
            f"toplam {total_samples:,} radar görüntüsü."
        )

    def get_image(self, frame: dict) -> np.ndarray:
        return self.chunks[frame["chunk_id"]][frame["local_idx"]]


def _style_panel(ax, title: str, bg: str | None = None):
    """Sol panel kutularına ortak kenarlık, dolgu ve başlık stili uygular."""
    C = THEME["colors"]
    FS = THEME["font_sizes"]
    ax.set_facecolor(bg or C["bg_panel"])
    ax.set_title(
        title,
        fontsize=FS["panel_title"],
        fontweight="bold",
        color=C["text"],
        loc="left",
        pad=6,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(C["border"])
        spine.set_linewidth(THEME["layout"]["panel_border_lw"])


def _style_button(btn: Button, bg: str | None = None, hover: str | None = None, bold: bool = False):
    """Alt şeritteki butonlara ortak tipografi ve kenarlık stili uygular."""
    C = THEME["colors"]
    btn.color = bg or C["bg_btn"]
    btn.hovercolor = hover or C["bg_btn_hover"]
    btn.ax.set_facecolor(btn.color)
    for spine in btn.ax.spines.values():
        spine.set_edgecolor(C["border"])
        spine.set_linewidth(THEME["layout"]["panel_border_lw"])
    btn.label.set_fontsize(THEME["font_sizes"]["body"])
    btn.label.set_color(C["text"])
    if bold:
        btn.label.set_fontweight("bold")


class EpisodeViewer:
    def __init__(self, dataset: EpisodeDataset):
        apply_theme()
        self.ds = dataset
        eps = dataset.episodes
        C = THEME["colors"]
        FS = THEME["font_sizes"]
        L = THEME["layout"]

        self.filters = {
            f"Tüm Fırtınalar ({len(eps):,})": [i for i in range(len(eps))],
            f"Hortumlu ({sum(e['has_tor'] for e in eps):,})": [
                i for i, e in enumerate(eps) if e["has_tor"]
            ],
            f"EF2+ Güçlü ({sum(e['max_ef'] >= 2 for e in eps):,})": [
                i for i, e in enumerate(eps) if e["max_ef"] >= 2
            ],
            f"Yanlış Alarm ({sum((not e['has_tor']) and e['has_wrn'] for e in eps):,})": [
                i for i, e in enumerate(eps) if (not e["has_tor"]) and e["has_wrn"]
            ],
            f"Normal / NUL ({sum((not e['has_tor']) and (not e['has_wrn']) for e in eps):,})": [
                i for i, e in enumerate(eps) if (not e["has_tor"]) and (not e["has_wrn"])
            ],
        }
        self.filter_names = list(self.filters.keys())
        self.active_filter = self.filter_names[1]  # Açılışta doğrudan hortumlu fırtınalarla başla
        if not self.filters[self.active_filter]:
            self.active_filter = self.filter_names[0]
        self.ep_pos = 0
        self.frame_idx = 0
        self.playing = False
        self._suppress_radio = False
        self.rng = np.random.default_rng()

        # Her kanal için NaN rengini ayıran colormap kopyaları
        self.cmaps = {}
        for v in self.ds.variables:
            self.cmaps[v] = plt.get_cmap(CMAPS[v]).with_extremes(bad=C["nan"])

        # Ana GridSpec düzeni: sol kenar çubuğu + sağda (üst başlık, 2x3 grafikler, zaman çizelgesi + kontrol şeridi)
        self.fig = plt.figure(figsize=L["figsize"], facecolor=C["bg_fig"])
        self.fig.canvas.manager.set_window_title("TorNet Radar Fırtına Gezgini")

        gs_root = self.fig.add_gridspec(
            3, 2,
            width_ratios=[0.21, 0.79],
            height_ratios=[0.115, 0.690, 0.195],
            left=0.018, right=0.982, top=0.955, bottom=0.025,
            wspace=0.135, hspace=0.220,
        )

        # --- Üst başlık bölgesi ---
        ax_header = self.fig.add_subplot(gs_root[0, 1])
        ax_header.set_axis_off()
        self.ax_header = ax_header
        self.status_badge = ax_header.text(
            0.5, 0.78, "", ha="center", va="center",
            fontsize=FS["badge"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", lw=1.4),
        )
        self.ep_title = ax_header.text(
            0.5, 0.34, "", ha="center", va="center",
            fontsize=FS["ep_title"], fontweight="bold", color=C["text"],
        )
        self.file_title = ax_header.text(
            0.5, 0.04, "", ha="center", va="center",
            fontsize=FS["filepath"], color=C["text_muted"],
        )

        # --- Sol panel (4 kutu ortak stille, içeriğe uyumlu yükseklik oranlarıyla) ---
        gs_left = gs_root[:, 0].subgridspec(4, 1, height_ratios=[1.12, 0.94, 0.56, 0.88], hspace=0.24)

        ax_radio = self.fig.add_subplot(gs_left[0, 0])
        _style_panel(ax_radio, "Fırtına Filtresi")
        self.radio = RadioButtons(ax_radio, self.filter_names, active=1)
        for lbl in self.radio.labels:
            lbl.set_fontsize(FS["body"])
            lbl.set_color(C["text"])
        self.radio.on_clicked(self.on_filter)

        ax_cat = self.fig.add_subplot(gs_left[1, 0])
        _style_panel(ax_cat, "Fırtına & Katalog")
        self.cat_text = ax_cat.text(
            0.06, 0.92, "", va="top", ha="left",
            fontsize=FS["mono"], family="monospace",
            linespacing=1.32, color=C["text"],
        )

        ax_stats = self.fig.add_subplot(gs_left[2, 0])
        _style_panel(ax_stats, "Radar Değerleri")
        self.stats_text = ax_stats.text(
            0.06, 0.88, "", va="top", ha="left",
            fontsize=FS["mono"], family="monospace",
            linespacing=1.45, color=C["text"],
        )

        ax_cursor = self.fig.add_subplot(gs_left[3, 0])
        _style_panel(ax_cursor, "İmleç (Senkron)", bg=C["bg_panel_alt"])
        self.cursor_text = ax_cursor.text(
            0.06, 0.90, "", va="top", ha="left",
            fontsize=FS["mono"], family="monospace",
            linespacing=1.30, color=C["text"],
        )

        # --- Orta alan: 2x3 grafikler (üst: CNN matrisi, alt: Gerçek dünya kutupsal görünüm) ---
        gs_plots = gs_root[1, 1].subgridspec(
            2, 3, height_ratios=[1.0, 1.06], wspace=0.34, hspace=0.25
        )
        self.axes_cnn = np.array([self.fig.add_subplot(gs_plots[0, c]) for c in range(3)])
        self.axes_world = np.array([self.fig.add_subplot(gs_plots[1, c]) for c in range(3)])
        self.axes = np.vstack([self.axes_cnn, self.axes_world])

        q0 = self.ds.get_image(eps[0]["frames"][0])
        phys0 = decode_physical_masked(q0, self.ds.variables)
        self.images = []
        for col, (ax_top, ax_bot, v, img) in enumerate(
            zip(self.axes_cnn, self.axes_world, self.ds.variables, phys0)
        ):
            lo, hi = RANGES[v]
            unit = UNITS[v]
            im = ax_top.imshow(img, aspect="auto", cmap=self.cmaps[v], vmin=lo, vmax=hi)
            ax_top.set_title(f"{v} ({unit})", fontsize=FS["plot_title"], fontweight="bold", pad=7)
            if col == 1:
                ax_top.set_xlabel("Mesafe (250 m / piksel)", fontsize=FS["axis_label"], labelpad=3)
            ax_top.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax_top.yaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax_top.tick_params(labelsize=FS["tick"], length=3)

            ax_bot.set_aspect("equal", adjustable="datalim")
            cbar = self.fig.colorbar(
                im, ax=[ax_top, ax_bot],
                fraction=L["cbar_fraction"], pad=L["cbar_pad"],
                shrink=L["cbar_shrink"], aspect=L["cbar_aspect"],
            )
            cbar.ax.set_title(unit, loc="left", x=0.0, fontsize=FS["axis_label"], pad=6, color=C["text_muted"])
            cbar.ax.tick_params(labelsize=FS["tick"], length=2.5)
            cbar.locator = MaxNLocator(5)
            cbar.update_ticks()
            cbar.outline.set_edgecolor(C["border"])
            cbar.outline.set_linewidth(0.8)
            self.images.append(im)

        self.axes_cnn[0].set_ylabel("CNN · Açı (0.5°/px)", fontsize=FS["axis_label"], labelpad=4)

        # --- Alt bölge: Zaman Çizelgesi + Gruplanmış Kontrol Şeridi ---
        gs_bottom = gs_root[2, 1].subgridspec(2, 1, height_ratios=[1.12, 0.88], hspace=0.58)

        self.ax_timeline = self.fig.add_subplot(gs_bottom[0, 0])
        self.ax_timeline.set_facecolor(C["bg_panel"])
        self.ax_timeline.set_yticks([])
        for spine in self.ax_timeline.spines.values():
            spine.set_edgecolor(C["border"])
            spine.set_linewidth(L["panel_border_lw"])

        # Gruplanmış kontrol butonları: Fırtına | Kare | Rastgele | Ep ID + Git | Oynat
        gs_btns = gs_bottom[1, 0].subgridspec(
            1, 12,
            width_ratios=[1.0, 1.0, 0.22, 0.95, 0.95, 0.22, 0.95, 0.22, 1.35, 0.65, 0.22, 1.05],
            wspace=0.08,
        )
        ax_ep_prev = self.fig.add_subplot(gs_btns[0, 0])
        ax_ep_next = self.fig.add_subplot(gs_btns[0, 1])
        ax_fr_prev = self.fig.add_subplot(gs_btns[0, 3])
        ax_fr_next = self.fig.add_subplot(gs_btns[0, 4])
        ax_rand = self.fig.add_subplot(gs_btns[0, 6])
        ax_box = self.fig.add_subplot(gs_btns[0, 8])
        ax_go = self.fig.add_subplot(gs_btns[0, 9])
        ax_play = self.fig.add_subplot(gs_btns[0, 11])

        # Gruplar arasına ince görsel ayırıcı çizgiler
        for sep_col in (2, 5, 7, 10):
            ax_sep = self.fig.add_subplot(gs_btns[0, sep_col])
            ax_sep.set_axis_off()
            ax_sep.plot([0.5, 0.5], [0.15, 0.85], color=C["border"], lw=1.2, transform=ax_sep.transAxes)

        self.btn_ep_prev = Button(ax_ep_prev, "◀ Fırtına")
        self.btn_ep_next = Button(ax_ep_next, "Fırtına ▶")
        self.btn_fr_prev = Button(ax_fr_prev, "◀ Kare")
        self.btn_fr_next = Button(ax_fr_next, "Kare ▶")
        self.btn_rand = Button(ax_rand, "Rastgele")
        self.box_ep = TextBox(ax_box, "Ep: ", initial="")
        self.btn_go = Button(ax_go, "Git")
        self.btn_play = Button(ax_play, "▶ Oynat")

        for b in (self.btn_ep_prev, self.btn_ep_next, self.btn_fr_prev, self.btn_fr_next, self.btn_rand, self.btn_go):
            _style_button(b)
        _style_button(self.btn_play, bg=C["bg_play"], hover=C["bg_play_hover"], bold=True)
        self.btn_play.label.set_color(C["accent"])

        self.box_ep.label.set_fontsize(FS["body"])
        self.box_ep.text_disp.set_fontsize(FS["body"])
        ax_box.set_facecolor(C["bg_panel"])
        for spine in ax_box.spines.values():
            spine.set_edgecolor(C["border"])
            spine.set_linewidth(L["panel_border_lw"])

        self.btn_ep_prev.on_clicked(lambda _: self.step_episode(-1))
        self.btn_ep_next.on_clicked(lambda _: self.step_episode(+1))
        self.btn_fr_prev.on_clicked(lambda _: self.step_frame(-1))
        self.btn_fr_next.on_clicked(lambda _: self.step_frame(+1))
        self.btn_rand.on_clicked(lambda _: self.jump_random())
        self.box_ep.on_submit(self.jump_to_text)
        self.btn_go.on_clicked(lambda _: self.jump_to_text(self.box_ep.text))
        self.btn_play.on_clicked(lambda _: self.toggle_play())

        self.current_fr: dict | None = None
        self.current_q: np.ndarray | None = None
        self.hover_cell: tuple[int, int] | None = None
        self.hover_markers: list = []

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("close_event", lambda _: self.stop_play())

        self.timer = self.fig.canvas.new_timer(interval=500)
        self.timer.add_callback(self._on_tick)

        self.render()
        plt.show()

    def current_pool(self) -> list[int]:
        return self.filters[self.active_filter]

    def current_episode(self) -> dict | None:
        pool = self.current_pool()
        if not pool:
            return None
        self.ep_pos %= len(pool)
        return self.ds.episodes[pool[self.ep_pos]]

    def _draw_timeline(self, ep: dict):
        ax = self.ax_timeline
        C = THEME["colors"]
        FS = THEME["font_sizes"]
        L = THEME["layout"]
        ax.clear()
        ax.set_facecolor(C["bg_panel"])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
            spine.set_linewidth(L["panel_border_lw"])

        frames = ep["frames"]
        n = len(frames)
        xs = np.arange(n)

        if n > 1:
            ax.plot(xs, np.zeros(n), color=C["timeline_line"], lw=1.4, zorder=1)

        # Geçmiş kareler hafif soluk, gelecek kareler normal, geçerli kare büyük ve belirgin
        for i, fr in enumerate(frames):
            c = C["warn"] if fr["y"] == 1 else C["safe"]
            m = "D" if fr["y"] == 1 else "o"
            if i == self.frame_idx:
                size, alpha, edge, lw, z = 110, 1.0, C["text"], 1.8, 4
            elif i < self.frame_idx:
                size, alpha, edge, lw, z = 24, 0.35, "none", 0.0, 2
            else:
                size, alpha, edge, lw, z = 28, 0.85, "#ffffff", 0.6, 3
            ax.scatter([i], [0], s=size, c=c, marker=m, alpha=alpha, edgecolors=edge, linewidths=lw, zorder=z)

        ax.axvline(self.frame_idx, color=C["text_muted"], ls="--", lw=1.0, alpha=0.65, zorder=2)

        # Eşit aralıklı en fazla 7 saat etiketi (çakışmayı önler)
        ticks = np.unique(np.round(np.linspace(0, n - 1, min(n, 7))).astype(int))
        ax.set_xticks(ticks)
        ax.set_xticklabels([frames[i]["end_time"][11:16] for i in ticks], fontsize=FS["tick"], color=C["text_muted"])
        ax.tick_params(axis="x", length=3, pad=2)
        ax.set_xlim(-0.6, max(0.6, n - 0.4))
        ax.set_ylim(-0.6, 0.6)

        ax.set_title(
            f"Zaman Çizelgesi · Kare {self.frame_idx + 1} / {n}",
            fontsize=FS["panel_title"] - 1.5, fontweight="bold", color=C["text"], loc="left", pad=5,
        )

        # Küçük SAFE / WARNING / NaN lejandı (başlıkla aynı satırda, sağda)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C["safe"], markersize=5.5, label="SAFE"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=C["warn"], markersize=5.5, label="WARNING"),
            Patch(facecolor=C["nan"], edgecolor=C["border"], linewidth=0.7, label="Veri yok (NaN)"),
        ]
        ax.legend(
            handles=legend_handles, loc="lower right", bbox_to_anchor=(1.0, 1.02),
            ncol=3, frameon=False, fontsize=FS["legend"], handletextpad=0.35, columnspacing=1.0,
        )

    def _clear_hover_markers(self):
        for artist in self.hover_markers:
            try:
                artist.remove()
            except Exception:
                pass
        self.hover_markers.clear()

    def _draw_hover_markers(self, fr: dict, cell: tuple[int, int]):
        self._clear_hover_markers()
        C = THEME["colors"]
        az_i, rng_i = cell
        az0 = fr["az_deg"][0]
        r0 = fr["rng_km"][0]
        az_deg = az0 + (az_i + 0.5) * 0.5
        r_km = r0 + (rng_i + 0.5) * 0.25

        for ax_top in self.axes_cnn:
            m_out = ax_top.plot(rng_i, az_i, marker="o", ms=9, mfc="none", mec=C["cursor_outer"], mew=2.4, zorder=10)[0]
            m_in = ax_top.plot(rng_i, az_i, marker="o", ms=9, mfc="none", mec=C["cursor_inner"], mew=1.4, zorder=11)[0]
            self.hover_markers.extend([m_out, m_in])

        if r_km >= 0.0:
            x_km = r_km * np.sin(np.deg2rad(az_deg))
            y_km = r_km * np.cos(np.deg2rad(az_deg))
            for ax_bot in self.axes_world:
                m_out = ax_bot.plot(x_km, y_km, marker="o", ms=9, mfc="none", mec=C["cursor_outer"], mew=2.4, zorder=10)[0]
                m_in = ax_bot.plot(x_km, y_km, marker="o", ms=9, mfc="none", mec=C["cursor_inner"], mew=1.4, zorder=11)[0]
                self.hover_markers.extend([m_out, m_in])

    def _update_cursor_box(self, fr: dict, q: np.ndarray, cell: tuple[int, int] | None):
        if cell is None:
            self.cursor_text.set_text(
                "Fareyi grafiklerin üzerine\ngetirin: tüm panellerde\naynı nokta işaretlenir."
            )
            return

        az_i, rng_i = cell
        az_deg = (fr["az_deg"][0] + (az_i + 0.5) * 0.5) % 360.0
        r_km = fr["rng_km"][0] + (rng_i + 0.5) * 0.25
        if r_km >= 0.0:
            x_km = r_km * np.sin(np.deg2rad(az_deg))
            y_km = r_km * np.cos(np.deg2rad(az_deg))
            loc_txt = f"Dünya : x={x_km:5.1f} y={y_km:5.1f}km"
        else:
            loc_txt = "Dünya : r<0 (sanal alan)"

        vals = []
        for i, v in enumerate(self.ds.variables):
            qv = int(q[i, az_i, rng_i])
            if qv == 0:
                vals.append(f"{v:5s}: veri yok (NaN)")
            else:
                lo, hi = RANGES[v]
                phys = (qv - 1.0) / 254.0 * (hi - lo) + lo
                vals.append(f"{v:5s}: {phys:6.1f} {UNITS[v]}")

        self.cursor_text.set_text(
            f"Piksel: (az={az_i}, rng={rng_i})\n"
            f"Konum : {az_deg:5.1f}° | {r_km:5.1f}km\n"
            f"{loc_txt}\n"
            + "\n".join(vals)
        )

    def _draw_radar_marker(self, ax, x_min: float, x_max: float, y_min: float, y_max: float):
        """Radar konumu (0, 0) görünür alandaysa işaretler; dışarıdaysa yönünü gösterir."""
        C = THEME["colors"]
        ax.apply_aspect()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if xlim[0] <= 0.0 <= xlim[1] and ylim[0] <= 0.0 <= ylim[1]:
            ax.scatter([0.0], [0.0], s=36, c=C["warn"], marker="^", edgecolors="#ffffff", linewidths=0.8, zorder=6)
            near_right = 0.0 > xlim[0] + 0.80 * (xlim[1] - xlim[0])
            ha = "right" if near_right else "left"
            lbl = "Radar  " if near_right else "  Radar"
            ax.text(0.0, 0.0, lbl, fontsize=7.5, color=C["warn"], fontweight="bold", va="center", ha=ha, zorder=6)
        else:
            cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
            ang = np.degrees(np.arctan2(-cy, -cx))
            arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
            arrow = arrows[int(round(ang / 45.0)) % 8]
            ax.text(
                0.97, 0.04, f"{arrow} Radar (0,0)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.3, color=C["text_muted"],
                bbox=dict(boxstyle="round,pad=0.2", fc=C["bg_panel"], ec=C["border_subtle"], alpha=0.9),
                zorder=6,
            )

    def render(self):
        C = THEME["colors"]
        FS = THEME["font_sizes"]
        L = THEME["layout"]

        ep = self.current_episode()
        if ep is None:
            self.current_fr = None
            self.current_q = None
            self._clear_hover_markers()
            self.status_badge.set_text("Seçili filtrede fırtına bulunamadı")
            self.status_badge.set_color(C["text_muted"])
            self.status_badge.get_bbox_patch().set(facecolor=C["bg_panel"], edgecolor=C["border"])
            self.ep_title.set_text(f"{self.active_filter} — 0 fırtına")
            self.file_title.set_text("")
            self.cat_text.set_text("-")
            self.stats_text.set_text("-")
            self.cursor_text.set_text("-")
            self.fig.canvas.draw_idle()
            return

        frames = ep["frames"]
        self.frame_idx %= len(frames)
        fr = frames[self.frame_idx]

        q = self.ds.get_image(fr)
        self.current_fr = fr
        self.current_q = q
        self._clear_hover_markers()

        phys_channels = decode_physical_masked(q, self.ds.variables)
        for im, img in zip(self.images, phys_channels):
            im.set_data(img)

        cat = fr["category"]
        y = fr["y"]
        ef_txt = f"EF{fr['ef']}" if fr["ef"] >= 0 else "Yok"

        if y == 1:
            accent_status, bg_color, edge_color = C["warn"], C["warn_bg"], C["warn_border"]
            badge_txt = f"TEHLİKELİ / WARNING (y=1)  ·  Hortum ({cat}, {ef_txt})"
        else:
            accent_status, bg_color, edge_color = C["safe"], C["safe_bg"], C["safe_border"]
            badge_txt = f"SAFE / GÜVENLİ (y=0)  ·  Hortum Yok (alt sınıf: {cat})"

        # Alt satır: her değişkenin gerçek dünya (kutupsal -> x, y km) görünümü
        az0, az1 = fr["az_deg"]
        r0, r1 = fr["rng_km"]
        start_idx = max(0, int(np.ceil(-r0 / 0.25)))
        has_world = start_idx < 240 and r1 > 0.0
        if has_world:
            r_pos_min = r0 + start_idx * 0.25
            az_edges = np.deg2rad(np.linspace(az0, az1, 121))
            rng_edges = np.linspace(r_pos_min, r1, 241 - start_idx)
            A, R = np.meshgrid(az_edges, rng_edges, indexing="ij")
            X_km, Y_km = R * np.sin(A), R * np.cos(A)
            border_poly = sector_polygon_xy(az0, az1, r_pos_min, r1)
            x_min, x_max = float(X_km.min()), float(X_km.max())
            y_min, y_max = float(Y_km.min()), float(Y_km.max())

        for col, (ax_bot, v, phys) in enumerate(zip(self.axes_world, self.ds.variables, phys_channels)):
            lo, hi = RANGES[v]
            ax_bot.clear()
            ax_bot.set_facecolor(C["bg_world"])
            if has_world:
                ax_bot.pcolormesh(X_km, Y_km, phys[:, start_idx:], cmap=self.cmaps[v], vmin=lo, vmax=hi, shading="flat")
                ax_bot.add_patch(Polygon(border_poly, closed=True, fill=False, ec=C["text"], lw=1.1, zorder=4))
                ax_bot.margins(0.06)
                ax_bot.set_aspect("equal", adjustable="datalim")
                ax_bot.xaxis.set_major_locator(MaxNLocator(4))
                ax_bot.yaxis.set_major_locator(MaxNLocator(4))
                ax_bot.tick_params(labelsize=FS["tick"], length=3)
                ax_bot.grid(color=C["grid_world"], linewidth=L["grid_world_lw"], alpha=L["grid_world_alpha"])
                self._draw_radar_marker(ax_bot, x_min, x_max, y_min, y_max)
            else:
                ax_bot.text(0.5, 0.5, "Görünür alan yok (r ≤ 0)", transform=ax_bot.transAxes, ha="center", va="center", color=C["text_muted"])
            if col == 1:
                ax_bot.set_xlabel("Radara göre doğu (km)", fontsize=FS["axis_label"], labelpad=3)
        self.axes_world[0].set_ylabel("Gerçek · Kuzey (km)", fontsize=FS["axis_label"], labelpad=4)

        for ax in self.axes.flat:
            for spine in ax.spines.values():
                spine.set_edgecolor(accent_status)
                spine.set_linewidth(L["plot_border_lw"])

        if self.hover_cell is not None:
            self._draw_hover_markers(fr, self.hover_cell)
        self._update_cursor_box(fr, q, self.hover_cell)

        self.status_badge.set_text(badge_txt)
        self.status_badge.set_color(accent_status)
        self.status_badge.get_bbox_patch().set(facecolor=bg_color, edgecolor=edge_color)

        pool = self.current_pool()
        self.ep_title.set_text(
            f"Fırtına #{self.ep_pos + 1:,} / {len(pool):,}  (Episode {ep['episode_id']})   ·   "
            f"Kare: {self.frame_idx + 1} / {ep['n_frames']}   ·   "
            f"Saat: {fr['start_time'][11:16]}–{fr['end_time'][11:16]} UTC"
        )
        self.file_title.set_text(fr["filename"])

        max_ef_txt = f"EF{ep['max_ef']}" if ep["max_ef"] >= 0 else "Yok"
        self.cat_text.set_text(
            f"Episode  : {ep['episode_id']} ({ep['split']})\n"
            f"Tarih    : {fr['start_time'][:10]}\n"
            f"İstasyon : {fr['site']} ({fr['lat']:.2f}, {fr['lon']:.2f})\n"
            f"Açı / Mes: {az0 % 360:.0f}°–{az1 % 360:.0f}° | {r0:.0f}–{r1:.0f}km\n"
            f"Şiddet   : {ef_txt} (max: {max_ef_txt})\n"
            f"Kareler  : {ep['n_frames']} (TOR={ep['n_tor']})\n"
            f"           WRN={ep['n_wrn']} | NUL={ep['n_nul']}"
        )
        self.stats_text.set_text(physical_stats(q, self.ds.variables))
        self._draw_timeline(ep)
        self.fig.canvas.draw_idle()

    def stop_play(self):
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.btn_play.label.set_text("▶ Oynat")

    def toggle_play(self):
        if self.playing:
            self.stop_play()
            self.fig.canvas.draw_idle()
        else:
            ep = self.current_episode()
            if ep is None or ep["n_frames"] <= 1:
                return
            self.playing = True
            self.btn_play.label.set_text("|| Durdur")
            self.timer.start()
            self.fig.canvas.draw_idle()

    def _on_tick(self):
        if not self.playing:
            return
        self.frame_idx += 1
        self.render()

    def step_episode(self, delta: int):
        self.stop_play()
        self.ep_pos += delta
        self.frame_idx = 0
        self.render()

    def step_frame(self, delta: int):
        self.stop_play()
        self.frame_idx += delta
        self.render()

    def jump_random(self):
        self.stop_play()
        pool = self.current_pool()
        if not pool:
            return
        self.ep_pos = int(self.rng.integers(0, len(pool)))
        self.frame_idx = 0
        self.render()

    def on_filter(self, label: str):
        if self._suppress_radio:
            return
        self.stop_play()
        self.active_filter = label
        self.ep_pos = 0
        self.frame_idx = 0
        self.render()

    def jump_to_text(self, text: str):
        val = text.strip()
        if not val:
            return
        self.stop_play()

        # Önce episode_id olarak ara
        if val in self.ds.ep_id_to_pos:
            global_ep = self.ds.ep_id_to_pos[val]
            pool = self.current_pool()
            if global_ep in pool:
                self.ep_pos = pool.index(global_ep)
            else:
                self.active_filter = self.filter_names[0]
                self._suppress_radio = True
                try:
                    self.radio.set_active(0)
                finally:
                    self._suppress_radio = False
                self.ep_pos = global_ep
            self.frame_idx = 0
            self.render()
            return

        # Değilse aktif filtredeki 1-tabanlı sıra numarası olarak atla
        try:
            num = int(val)
        except ValueError:
            return
        pool = self.current_pool()
        if not pool:
            return
        self.ep_pos = max(0, min(len(pool) - 1, num - 1))
        self.frame_idx = 0
        self.render()

    def on_click(self, event):
        if event.inaxes is self.ax_timeline and event.xdata is not None:
            self.stop_play()
            ep = self.current_episode()
            if ep is None:
                return
            n = ep["n_frames"]
            idx = int(round(event.xdata))
            self.frame_idx = max(0, min(n - 1, idx))
            self.render()

    def on_motion(self, event):
        if self.current_fr is None or self.current_q is None:
            return
        fr = self.current_fr
        cell: tuple[int, int] | None = None

        if any(event.inaxes is ax for ax in self.axes_cnn) and event.xdata is not None and event.ydata is not None:
            rng_i = int(round(event.xdata))
            az_i = int(round(event.ydata))
            if 0 <= az_i < 120 and 0 <= rng_i < 240:
                cell = (az_i, rng_i)
        elif any(event.inaxes is ax for ax in self.axes_world) and event.xdata is not None and event.ydata is not None:
            r_km = float(np.hypot(event.xdata, event.ydata))
            az_deg = float(np.degrees(np.arctan2(event.xdata, event.ydata)) % 360.0)
            az0 = fr["az_deg"][0]
            r0, r1 = fr["rng_km"]
            d_az = (az_deg - (az0 % 360.0)) % 360.0
            if 0.0 <= d_az < 60.0 and r0 <= r_km < r1:
                az_i = min(119, max(0, int(d_az / 0.5)))
                rng_i = min(239, max(0, int((r_km - r0) / 0.25)))
                cell = (az_i, rng_i)

        if cell == self.hover_cell:
            return
        self.hover_cell = cell
        if cell is None:
            self._clear_hover_markers()
        else:
            self._draw_hover_markers(fr, cell)
        self._update_cursor_box(fr, self.current_q, cell)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if getattr(self.box_ep, "capturekeystrokes", False):
            return
        if event.key == "left":
            self.step_frame(-1)
        elif event.key == "right":
            self.step_frame(+1)
        elif event.key in ("up", "["):
            self.step_episode(-1)
        elif event.key in ("down", "]"):
            self.step_episode(+1)
        elif event.key == " ":
            self.toggle_play()
        elif event.key == "r":
            self.jump_random()


if __name__ == "__main__":
    dataset = EpisodeDataset(find_npz_files(), find_catalog())
    EpisodeViewer(dataset)
