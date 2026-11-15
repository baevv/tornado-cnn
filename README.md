# tornado-cnn

Tornado detection from weather radar data with convolutional neural networks.
Course project for *Artificial Neural Networks* (Yapay Sinir Ağları).

Given a radar image of a storm, the model predicts whether a tornado is on the ground.
The data is [TorNet](https://github.com/mit-ll/tornet), a public benchmark of US weather radar (NEXRAD) samples.

## Data

| | |
|---|---|
| Samples | 203,133 (2013–2022), ~170 GB |
| Tornado samples | 13,857 (6.8%), strongly imbalanced |
| One sample | 6 radar variables × 4 frames (5 min apart) × 2 elevation angles × 120 azimuth × 240 range |
| Used here | 2015, 2017, 2019: 68,314 samples, 5,388 tornadoes |
| Input to the network | reflectivity (DBZ) + radial velocity (VEL), lowest sweep, last frame → `2 × 120 × 240` |

Labels: `TOR` (confirmed tornado) = 1; `WRN` (warning issued, no tornado) and `NUL` (random storm) = 0.
The official test split of each year is kept untouched for the final evaluation.

The data is **not** stored in this repository. The notebooks download it from Zenodo.

## Project structure

```
notebooks/
  01_explore_data.ipynb   catalog statistics, radar images, preprocessing check
  02_preprocess.ipynb     download selected years, save compact arrays to Google Drive
src/
  paths.py                storage locations on Colab / locally
  tornet.py               download, catalog, reading and preprocessing samples
  viz.py                  radar plots
configs/
  data.json               years, channels, sweep, frame
results/                  figures and metric tables
```

## Running on Google Colab

1. **GitHub token (once).** GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token.
   Repository access: *Only select repositories* → `tornado-cnn`. Permissions: *Contents → Read-only*.
2. **Store it in Colab (once).** In Colab, click the key icon (Secrets) on the left, add `GITHUB_TOKEN` with the token as value and enable *Notebook access*.
3. **Open a notebook.** File → Open notebook → GitHub tab → tick *Include private repos* → choose a notebook from this repo.
4. Run the first cell. It mounts Google Drive, clones the repo and installs missing packages.

Never paste the token into a notebook or commit it.

Preprocessing (`02_preprocess.ipynb`) needs no GPU; use a CPU runtime. Training will use a GPU runtime.

## Running locally (macOS / Linux)

```bash
git clone https://github.com/baevv/tornado-cnn.git
cd tornado-cnn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open the notebooks in VS Code (Python + Jupyter extensions) and select the `.venv` kernel.
By default data goes to `data/` inside the repo (ignored by git). To use the processed files that Colab saved to Google Drive,
point `TORNADO_DATA_DIR` at the synced Drive folder, e.g.

```bash
export TORNADO_DATA_DIR="$HOME/Library/CloudStorage/GoogleDrive-<account>/My Drive/tornado-cnn/data"
```

## Roadmap

- [x] Data exploration and preprocessing pipeline
- [ ] Train/validation split by storm episode, data loader
- [ ] CNN baseline with class-weighted loss
- [ ] Optimizer comparison (SGD, Momentum, RMSProp, Adam)
- [ ] Regularization (Dropout, L2), learning curve
- [ ] Evaluation on the test split (ROC-AUC, PR-AUC, F1, confusion matrix)
- [ ] Grad-CAM explanations

## Citation and license

Data: TorNet, MIT Lincoln Laboratory, licensed under CC BY 4.0
([Zenodo](https://doi.org/10.5281/zenodo.12636522)).

> Veillette, M. S., et al. (2025). A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution
> Polarimetric Weather Radar Data. *Artificial Intelligence for the Earth Systems*, 4(1).
