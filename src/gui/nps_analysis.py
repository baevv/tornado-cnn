from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

RANGES = {"DBZ": (-20.0, 60.0), "VEL": (-60.0, 60.0), "WIDTH": (0.0, 16.0)}
FILL_NORM = {"DBZ": -1.0, "VEL": 0.0, "WIDTH": -1.0}
CMAPS = {"DBZ": "turbo", "VEL": "seismic", "WIDTH": "magma"}


def find_npz_files() -> list[Path]:
    """Önce ~/tor_npz klasörüne, orada yoksa ~ (ev dizini) içine bakar."""
    for folder in (Path.home() / "tor_npz", Path.home()):
        files = sorted(folder.glob("tornet_*.npz"))
        if files:
            return files
    return []


def decode_for_model(q: np.ndarray, variables: list[str]) -> np.ndarray:
    """uint8 (3, 120, 240) -> [-1, 1] float32."""
    x = (q.astype(np.float32) - 1.0) / 254.0 * 2.0 - 1.0
    for i, v in enumerate(variables):
        x[i][q[i] == 0] = FILL_NORM.get(v, -1.0)
    return x


UNITS = {"DBZ": "dBZ", "VEL": "m/s", "WIDTH": "m/s"}


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


class UnifiedTorNet:
    """Birden fazla .npz dosyasını birleştirmeden tek indeks (0..N-1) ile sunar."""

    def __init__(self, paths: list[Path]):
        if not paths:
            raise FileNotFoundError("tornet_*.npz dosyası bulunamadı.")
        self.chunks = []
        self.offsets = [0]
        y_all, name_all, year_all = [], [], []
        self.variables = None

        for p in paths:
            print(f"Yükleniyor: {p} ...")
            with np.load(p) as d:
                X = d["X"]
                y = d["y"]
                names = np.array([str(n) for n in d["filename"]])
                if self.variables is None:
                    self.variables = [str(v) for v in d["variables"]]
            year = p.stem.replace("tornet_", "")
            self.chunks.append((year, X, y, names))
            self.offsets.append(self.offsets[-1] + len(y))
            y_all.append(y)
            name_all.append(names)
            year_all.append(np.full(len(y), year))

        self.total = self.offsets[-1]
        self.y = np.concatenate(y_all)
        self.names = np.concatenate(name_all)
        self.years = np.concatenate(year_all)
        self.cats = np.array([Path(n).name[:3] for n in self.names])
        print(f"Hazır: toplam {self.total:,} örnek ({int(self.y.sum()):,} hortum).")

    def get(self, global_idx: int):
        chunk_id = int(np.searchsorted(self.offsets[1:], global_idx, side="right"))
        local_idx = global_idx - self.offsets[chunk_id]
        year, X, y, names = self.chunks[chunk_id]
        return year, X[local_idx], int(y[local_idx]), str(names[local_idx])


class TorNetViewer:
    def __init__(self, dataset: UnifiedTorNet):
        self.ds = dataset
        self.filter_indices = {
            f"Tümü ({dataset.total:,})": np.arange(dataset.total),
            f"WARNING ({int((dataset.y == 1).sum()):,})": np.where(dataset.y == 1)[0],
            f"SAFE ({int((dataset.y == 0).sum()):,})": np.where(dataset.y == 0)[0],
        }
        self.filter_names = list(self.filter_indices.keys())
        self.active_filter = self.filter_names[0]
        self.pos = 0
        self.rng = np.random.default_rng()

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16.0, 6.2))
        self.fig.subplots_adjust(left=0.22, right=0.955, top=0.80, bottom=0.19, wspace=0.28)

        # İlk görselleri oluştur
        _, q0, _, _ = self.ds.get(0)
        x0 = decode_for_model(q0, self.ds.variables)
        self.images = []
        for ax, v, img in zip(self.axes, self.ds.variables, x0):
            im = ax.imshow(img, aspect="auto", cmap=CMAPS[v], vmin=-1, vmax=1)
            ax.set_title(v, fontsize=11, fontweight="bold")
            ax.set_xlabel("mesafe (250 m / piksel)")
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self.images.append(im)
        self.axes[0].set_ylabel("açı (0.5° / piksel)")

        self.status_badge = self.fig.text(
            0.585, 0.94, "", ha="center", va="center", fontsize=11.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.38", lw=2),
        )
        self.title_text = self.fig.text(0.585, 0.875, "", ha="center", va="center", fontsize=10.5)

        # Sol panel: Filtre kutusu + altında satır satır değer kutusu
        ax_radio = self.fig.add_axes([0.015, 0.54, 0.145, 0.25])
        ax_radio.set_title("Filtre", fontsize=10, fontweight="bold")
        self.radio = RadioButtons(ax_radio, self.filter_names)
        self.radio.on_clicked(self.on_filter)

        ax_stats = self.fig.add_axes([0.015, 0.19, 0.145, 0.29], facecolor="#f8f9fa")
        ax_stats.set_title("Değerler", fontsize=10, fontweight="bold")
        ax_stats.set_xticks([])
        ax_stats.set_yticks([])
        self.stats_text = ax_stats.text(
            0.06, 0.92, "", va="top", ha="left", fontsize=8.5, family="monospace"
        )

        # Alt kontrol çubuğu
        ax_prev = self.fig.add_axes([0.25, 0.04, 0.10, 0.06])
        ax_next = self.fig.add_axes([0.36, 0.04, 0.10, 0.06])
        ax_rand = self.fig.add_axes([0.48, 0.04, 0.10, 0.06])
        ax_box = self.fig.add_axes([0.69, 0.04, 0.11, 0.06])
        ax_go = self.fig.add_axes([0.81, 0.04, 0.08, 0.06])

        self.btn_prev = Button(ax_prev, "◀ Önceki")
        self.btn_next = Button(ax_next, "Sonraki ▶")
        self.btn_rand = Button(ax_rand, "Rastgele")
        self.box_idx = TextBox(ax_box, "Örnek No: ", initial="0")
        self.btn_go = Button(ax_go, "Git")

        self.btn_prev.on_clicked(lambda _: self.step(-1))
        self.btn_next.on_clicked(lambda _: self.step(+1))
        self.btn_rand.on_clicked(lambda _: self.jump_random())
        self.box_idx.on_submit(self.jump_to_text)
        self.btn_go.on_clicked(lambda _: self.jump_to_text(self.box_idx.text))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.render()
        plt.show()

    def current_pool(self) -> np.ndarray:
        return self.filter_indices[self.active_filter]

    def render(self):
        pool = self.current_pool()
        self.pos %= len(pool)
        global_idx = int(pool[self.pos])

        year, q, y, name = self.ds.get(global_idx)
        x = decode_for_model(q, self.ds.variables)
        for im, img in zip(self.images, x):
            im.set_data(img)

        cat = Path(name).name[:3]
        if y == 1:
            border_color, bg_color = "#d32f2f", "#fdecea"
            badge_txt = f"TEHLİKELİ / WARNING (y=1)  ·  Doğrulanmış Hortum ({cat})"
        else:
            border_color, bg_color = "#2e7d32", "#e8f5e9"
            badge_txt = f"SAFE / GÜVENLİ (y=0)  ·  Hortum Yok (alt sınıf: {cat})"

        for ax in self.axes:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3.0)

        self.status_badge.set_text(badge_txt)
        self.status_badge.set_color(border_color)
        self.status_badge.get_bbox_patch().set(facecolor=bg_color, edgecolor=border_color)

        self.title_text.set_text(
            f"Genel #{global_idx:,} / {self.ds.total - 1:,}  "
            f"(Filtre sırası: {self.pos + 1:,}/{len(pool):,})   |   "
            f"Yıl: {year}   |   {name}"
        )
        self.stats_text.set_text(physical_stats(q, self.ds.variables))
        self.fig.canvas.draw_idle()

    def step(self, delta: int):
        self.pos += delta
        self.render()

    def jump_random(self):
        pool = self.current_pool()
        self.pos = int(self.rng.integers(0, len(pool)))
        self.render()

    def on_filter(self, label: str):
        self.active_filter = label
        self.pos = 0
        self.render()

    def jump_to_text(self, text: str):
        try:
            idx = int(text.strip())
        except ValueError:
            return
        idx = max(0, min(self.ds.total - 1, idx))
        pool = self.current_pool()
        hit = np.where(pool == idx)[0]
        if len(hit) > 0:
            self.pos = int(hit[0])
        else:
            # Girilen numara aktif filtrede yoksa Tümü'ne geçip orada açar
            self.active_filter = self.filter_names[0]
            self.radio.set_active(0)
            self.pos = idx
        self.render()

    def on_key(self, event):
        if event.key == "left":
            self.step(-1)
        elif event.key == "right":
            self.step(+1)
        elif event.key == "r":
            self.jump_random()


if __name__ == "__main__":
    ds = UnifiedTorNet(find_npz_files())
    TorNetViewer(ds)