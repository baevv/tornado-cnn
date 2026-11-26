"""Fırtına olayı (episode_id) bazlı TorNet .npz + catalog.csv görselleştirici.

Özellikler:
  * ~/tor_npz (veya ~ / data/processed) içindeki tornet_*.npz dosyalarını RAM'de
    kopyalamadan yükler ve catalog.csv ile filename üzerinden eşleştirir.
  * Örnekleri episode_id altında zaman sırasına (start_time) göre gruplar.
  * Üst seviyede fırtına olayları arasında, alt seviyede ise seçili fırtınanın
    tüm zaman kareleri arasında gezinme + Oynat/Durdur + tıklanabilir zaman çizelgesi sunar.
  * Üst satırda CNN'in gördüğü 120×240 ham matrisi, alt satırda ise her değişkenin
    insan gözüyle anlaşılabilen gerçek dünya (kutupsal -> x, y km) görünümünü çizer.
  * Etiket y=1 (TOR) için kırmızı çerçeve, y=0 (WRN + NUL) için yeşil çerçeve gösterir.
"""
from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RANGES = {"DBZ": (-20.0, 60.0), "VEL": (-60.0, 60.0), "WIDTH": (0.0, 16.0)}
FILL_NORM = {"DBZ": -1.0, "VEL": 0.0, "WIDTH": -1.0}
CMAPS = {"DBZ": "turbo", "VEL": "seismic", "WIDTH": "magma"}
UNITS = {"DBZ": "dBZ", "VEL": "m/s", "WIDTH": "m/s"}


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


def sector_polygon_xy(az_a: float, az_b: float, r_a: float, r_b: float, n: int = 48) -> np.ndarray:
    """Kutupsal (az_a..az_b derece, r_a..r_b km) diliminin kapalı (x, y) poligon köşeleri."""
    az1 = np.deg2rad(np.linspace(az_a, az_b, n))
    az2 = np.deg2rad(np.linspace(az_b, az_a, n))
    x = np.concatenate([r_a * np.sin(az1), r_b * np.sin(az2)])
    y = np.concatenate([r_a * np.cos(az1), r_b * np.cos(az2)])
    return np.column_stack([x, y])


def physical_stats(q: np.ndarray, variables: list[str]) -> str:
    """Kanal başına fiziksel min/max ve boş piksel yüzdesini satır satır biçimlendirir."""
    blocks = []
    for i, v in enumerate(variables):
        lo, hi = RANGES[v]
        ok = q[i] > 0
        empty = 100.0 * float((~ok).mean())
        unit = UNITS.get(v, "")
        if ok.any():
            vals = (q[i][ok].astype(np.float32) - 1.0) / 254.0 * (hi - lo) + lo
            line = f"[{vals.min():5.1f}, {vals.max():5.1f}] | boş %{empty:2.0f}"
        else:
            line = "tamamen boş"
        blocks.append(f"{v} ({unit})\n  {line}")
    return "\n\n".join(blocks)


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


class EpisodeViewer:
    def __init__(self, dataset: EpisodeDataset):
        self.ds = dataset
        eps = dataset.episodes

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
        self.ep_pos = 0
        self.frame_idx = 0
        self.playing = False
        self.rng = np.random.default_rng()

        self.fig, self.axes = plt.subplots(
            2, 3, figsize=(16.8, 9.4), gridspec_kw={"height_ratios": [1.0, 1.12]}
        )
        self.fig.subplots_adjust(left=0.24, right=0.955, top=0.83, bottom=0.225, wspace=0.26, hspace=0.24)
        self.axes_cnn = self.axes[0]
        self.axes_world = self.axes[1]

        # İlk placeholder görüntüleri (Üst: CNN 120x240, Alt: Gerçek dünya kutupsal x/y km)
        q0 = self.ds.get_image(eps[0]["frames"][0])
        x0 = decode_for_model(q0, self.ds.variables)
        self.images = []
        for ax_top, ax_bot, v, img in zip(self.axes_cnn, self.axes_world, self.ds.variables, x0):
            im = ax_top.imshow(img, aspect="auto", cmap=CMAPS[v], vmin=-1, vmax=1)
            ax_top.set_title(v, fontsize=11, fontweight="bold")
            ax_top.set_xlabel("mesafe (250 m / piksel)", fontsize=8.8)
            ax_bot.set_aspect("equal", adjustable="datalim")
            self.fig.colorbar(im, ax=[ax_top, ax_bot], fraction=0.046, pad=0.035)
            self.images.append(im)
        self.axes_cnn[0].set_ylabel("CNN Girdisi\naçı (0.5° / piksel)", fontsize=9.2)

        # Üst başlık ve kırmızı/yeşil rozet
        self.status_badge = self.fig.text(
            0.585, 0.958, "", ha="center", va="center", fontsize=11.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.36", lw=2),
        )
        self.ep_title = self.fig.text(0.585, 0.908, "", ha="center", va="center", fontsize=10.5, fontweight="bold")
        self.file_title = self.fig.text(0.585, 0.870, "", ha="center", va="center", fontsize=9.0, color="#444444")

        # Sol panel: 1) Filtre, 2) Katalog/Fırtına Bilgisi, 3) Radar Değerleri, 4) Senkron İmleç
        ax_radio = self.fig.add_axes([0.015, 0.675, 0.172, 0.175])
        ax_radio.set_title("Fırtına Filtresi", fontsize=9.8, fontweight="bold")
        self.radio = RadioButtons(ax_radio, self.filter_names, active=1)
        self.radio.on_clicked(self.on_filter)

        ax_cat = self.fig.add_axes([0.015, 0.415, 0.172, 0.225], facecolor="#f4f6f8")
        ax_cat.set_title("Fırtına & Katalog", fontsize=9.8, fontweight="bold")
        ax_cat.set_xticks([])
        ax_cat.set_yticks([])
        self.cat_text = ax_cat.text(
            0.05, 0.94, "", va="top", ha="left", fontsize=8.1, family="monospace"
        )

        ax_stats = self.fig.add_axes([0.015, 0.205, 0.172, 0.175], facecolor="#f8f9fa")
        ax_stats.set_title("Radar Değerleri", fontsize=9.8, fontweight="bold")
        ax_stats.set_xticks([])
        ax_stats.set_yticks([])
        self.stats_text = ax_stats.text(
            0.05, 0.92, "", va="top", ha="left", fontsize=8.0, family="monospace"
        )

        ax_cursor = self.fig.add_axes([0.015, 0.025, 0.172, 0.145], facecolor="#eef6ff")
        ax_cursor.set_title("İmleç (Senkron)", fontsize=9.8, fontweight="bold")
        ax_cursor.set_xticks([])
        ax_cursor.set_yticks([])
        self.cursor_text = ax_cursor.text(
            0.05, 0.92, "", va="top", ha="left", fontsize=8.0, family="monospace"
        )

        # Zaman çizelgesi (fırtınanın tüm kareleri; tıklanabilir)
        self.ax_timeline = self.fig.add_axes([0.24, 0.102, 0.56, 0.055], facecolor="#fafafa")
        self.ax_timeline.set_title(
            "Fırtınanın Zaman Çizelgesi (kareye tıklayarak atlayabilirsin: yeşil=SAFE, kırmızı=WARNING)",
            fontsize=9.0, loc="left",
        )
        self.ax_timeline.set_yticks([])

        ax_play = self.fig.add_axes([0.82, 0.104, 0.11, 0.050])
        self.btn_play = Button(ax_play, "▶ Oynat")
        self.btn_play.on_clicked(lambda _: self.toggle_play())

        # Alt kontrol çubuğu
        ax_ep_prev = self.fig.add_axes([0.24, 0.025, 0.095, 0.048])
        ax_ep_next = self.fig.add_axes([0.34, 0.025, 0.095, 0.048])
        ax_fr_prev = self.fig.add_axes([0.45, 0.025, 0.085, 0.048])
        ax_fr_next = self.fig.add_axes([0.54, 0.025, 0.085, 0.048])
        ax_rand = self.fig.add_axes([0.64, 0.025, 0.08, 0.048])
        ax_box = self.fig.add_axes([0.80, 0.025, 0.075, 0.048])
        ax_go = self.fig.add_axes([0.88, 0.025, 0.05, 0.048])

        self.btn_ep_prev = Button(ax_ep_prev, "◀ Fırtına")
        self.btn_ep_next = Button(ax_ep_next, "Fırtına ▶")
        self.btn_fr_prev = Button(ax_fr_prev, "◀ Kare")
        self.btn_fr_next = Button(ax_fr_next, "Kare ▶")
        self.btn_rand = Button(ax_rand, "Rastgele")
        self.box_ep = TextBox(ax_box, "Ep ID/No: ", initial="")
        self.btn_go = Button(ax_go, "Git")

        self.btn_ep_prev.on_clicked(lambda _: self.step_episode(-1))
        self.btn_ep_next.on_clicked(lambda _: self.step_episode(+1))
        self.btn_fr_prev.on_clicked(lambda _: self.step_frame(-1))
        self.btn_fr_next.on_clicked(lambda _: self.step_frame(+1))
        self.btn_rand.on_clicked(lambda _: self.jump_random())
        self.box_ep.on_submit(self.jump_to_text)
        self.btn_go.on_clicked(lambda _: self.jump_to_text(self.box_ep.text))

        self.current_fr: dict | None = None
        self.current_q: np.ndarray | None = None
        self.hover_cell: tuple[int, int] | None = None
        self.hover_markers: list = []

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.timer = self.fig.canvas.new_timer(interval=500)
        self.timer.add_callback(self._on_tick)

        self.render()
        plt.show()

    def current_pool(self) -> list[int]:
        return self.filters[self.active_filter]

    def current_episode(self) -> dict:
        pool = self.current_pool()
        self.ep_pos %= len(pool)
        return self.ds.episodes[pool[self.ep_pos]]

    def _draw_timeline(self, ep: dict):
        ax = self.ax_timeline
        ax.clear()
        ax.set_yticks([])
        frames = ep["frames"]
        n = len(frames)

        xs = np.arange(n)
        colors = ["#d32f2f" if fr["y"] == 1 else "#2e7d32" for fr in frames]
        markers = ["D" if fr["y"] == 1 else "o" for fr in frames]

        if n > 1:
            ax.plot(xs, np.zeros(n), color="#cccccc", lw=2, zorder=1)
        for i, (x, c, m) in enumerate(zip(xs, colors, markers)):
            size = 110 if i == self.frame_idx else 60
            edge = "black" if i == self.frame_idx else "white"
            lw = 2.2 if i == self.frame_idx else 0.8
            ax.scatter([x], [0], s=size, c=c, marker=m, edgecolors=edge, linewidths=lw, zorder=3)

        ax.axvline(self.frame_idx, color="#222222", ls="--", lw=1.2, alpha=0.6, zorder=2)

        # Zaman etiketleri (kalabalık olmaması için en fazla ~10 etiket göster)
        step = max(1, (n + 9) // 10)
        ticks = list(range(0, n, step))
        if (n - 1) not in ticks:
            ticks.append(n - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([frames[i]["end_time"][11:16] for i in ticks], fontsize=8)
        ax.set_xlim(-0.6, max(0.6, n - 0.4))
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(
            f"Fırtınanın Zaman Çizelgesi — Kare {self.frame_idx + 1}/{n} "
            f"(tıklayarak atla | yeşil=SAFE, kırmızı=WARNING)",
            fontsize=9.0, loc="left",
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
        az_i, rng_i = cell
        az0 = fr["az_deg"][0]
        r0 = fr["rng_km"][0]
        az_deg = az0 + (az_i + 0.5) * 0.5
        r_km = r0 + (rng_i + 0.5) * 0.25

        for ax_top in self.axes_cnn:
            m_out = ax_top.plot(rng_i, az_i, marker="o", ms=9, mfc="none", mec="#000000", mew=2.4, zorder=10)[0]
            m_in = ax_top.plot(rng_i, az_i, marker="o", ms=9, mfc="none", mec="#00e5ff", mew=1.4, zorder=11)[0]
            self.hover_markers.extend([m_out, m_in])

        if r_km >= 0.0:
            x_km = r_km * np.sin(np.deg2rad(az_deg))
            y_km = r_km * np.cos(np.deg2rad(az_deg))
            for ax_bot in self.axes_world:
                m_out = ax_bot.plot(x_km, y_km, marker="o", ms=9, mfc="none", mec="#000000", mew=2.4, zorder=10)[0]
                m_in = ax_bot.plot(x_km, y_km, marker="o", ms=9, mfc="none", mec="#00e5ff", mew=1.4, zorder=11)[0]
                self.hover_markers.extend([m_out, m_in])

    def _update_cursor_box(self, fr: dict, q: np.ndarray, cell: tuple[int, int] | None):
        if cell is None:
            self.cursor_text.set_text(
                "Fareyi 6 grafikten\nbirinin üzerine getirin:\ntümünde aynı nokta\nişaretlenir."
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
                vals.append(f"{v:5s}: boş (NaN)")
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

    def render(self):
        ep = self.current_episode()
        frames = ep["frames"]
        self.frame_idx %= len(frames)
        fr = frames[self.frame_idx]

        q = self.ds.get_image(fr)
        self.current_fr = fr
        self.current_q = q
        self._clear_hover_markers()
        x = decode_for_model(q, self.ds.variables)
        for im, img in zip(self.images, x):
            im.set_data(img)

        cat = fr["category"]
        y = fr["y"]
        ef_txt = f"EF{fr['ef']}" if fr["ef"] >= 0 else "Yok"

        if y == 1:
            border_color, bg_color = "#d32f2f", "#fdecea"
            badge_txt = f"TEHLİKELİ / WARNING (y=1)  ·  Hortum ({cat}, {ef_txt})"
        else:
            border_color, bg_color = "#2e7d32", "#e8f5e9"
            badge_txt = f"SAFE / GÜVENLİ (y=0)  ·  Hortum Yok (alt sınıf: {cat})"

        # Alt satır: her değişkenin gerçek dünya (kutupsal -> x, y km) görünümü
        az0, az1 = fr["az_deg"]
        r0, r1 = fr["rng_km"]
        start_idx = max(0, int(np.ceil(-r0 / 0.25)))
        r_pos_min = r0 + start_idx * 0.25
        az_edges = np.deg2rad(np.linspace(az0, az1, 121))
        rng_edges = np.linspace(r_pos_min, r1, 241 - start_idx)
        A, R = np.meshgrid(az_edges, rng_edges, indexing="ij")
        X_km, Y_km = R * np.sin(A), R * np.cos(A)
        border_poly = sector_polygon_xy(az0, az1, r_pos_min, r1)

        for i, (ax_bot, v) in enumerate(zip(self.axes_world, self.ds.variables)):
            ax_bot.clear()
            ax_bot.set_facecolor("#f0f2f5")
            ax_bot.pcolormesh(X_km, Y_km, x[i, :, start_idx:], cmap=CMAPS[v], vmin=-1, vmax=1, shading="flat")
            ax_bot.add_patch(Polygon(border_poly, closed=True, fill=False, ec="#222222", lw=1.4, zorder=4))
            ax_bot.margins(0.05)
            ax_bot.set_aspect("equal", adjustable="datalim")
            ax_bot.set_xlabel("radara göre doğu (km)", fontsize=8.8)
            ax_bot.tick_params(labelsize=8.0)
            ax_bot.grid(alpha=0.25)
        self.axes_world[0].set_ylabel("Gerçek Dünya\nradara göre kuzey (km)", fontsize=9.2)

        for ax in self.axes.flat:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3.0)

        if self.hover_cell is not None:
            self._draw_hover_markers(fr, self.hover_cell)
        self._update_cursor_box(fr, q, self.hover_cell)

        self.status_badge.set_text(badge_txt)
        self.status_badge.set_color(border_color)
        self.status_badge.get_bbox_patch().set(facecolor=bg_color, edgecolor=border_color)

        pool = self.current_pool()
        self.ep_title.set_text(
            f"Fırtına #{self.ep_pos + 1:,} / {len(pool):,}  (Episode {ep['episode_id']})   |   "
            f"Kare: {self.frame_idx + 1} / {ep['n_frames']}   |   "
            f"Saat: {fr['start_time'][11:16]}–{fr['end_time'][11:16]} UTC"
        )
        self.file_title.set_text(fr["filename"])

        max_ef_txt = f"EF{ep['max_ef']}" if ep["max_ef"] >= 0 else "Yok"
        self.cat_text.set_text(
            f"Episode ID : {ep['episode_id']}\n"
            f"Küme (split): {ep['split']}\n"
            f"Tarih      : {fr['start_time'][:10]}\n"
            f"İstasyon   : {fr['site']} ({fr['lat']:.2f}, {fr['lon']:.2f})\n"
            f"Açı        : {az0 % 360:.0f}°–{az1 % 360:.0f}° (60°)\n"
            f"Uzaklık    : {r0:.0f}–{r1:.0f} km\n"
            f"Kare Şiddet: {ef_txt} (Fırtına max: {max_ef_txt})\n"
            f"Fırtına Özeti ({ep['n_frames']} kare):\n"
            f"  TOR={ep['n_tor']} | WRN={ep['n_wrn']} | NUL={ep['n_nul']}"
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
            if self.current_episode()["n_frames"] <= 1:
                return
            self.playing = True
            self.btn_play.label.set_text("⏸ Durdur")
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
        self.ep_pos = int(self.rng.integers(0, len(pool)))
        self.frame_idx = 0
        self.render()

    def on_filter(self, label: str):
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
                self.radio.set_active(0)
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
        self.ep_pos = max(0, min(len(pool) - 1, num - 1))
        self.frame_idx = 0
        self.render()

    def on_click(self, event):
        if event.inaxes is self.ax_timeline and event.xdata is not None:
            self.stop_play()
            n = self.current_episode()["n_frames"]
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
