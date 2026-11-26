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
| Input to the network | reflectivity (DBZ) + radial velocity (VEL) + spectrum width (WIDTH), lowest sweep, last frame → `3 × 120 × 240` (channels chosen by the EDA in notebook 01) |

Labels: `TOR` (confirmed tornado) = 1; `WRN` (warning issued, no tornado) and `NUL` (random storm) = 0.
The official test split of each year is kept untouched for the final evaluation.

The data is **not** stored in this repository. The notebooks download it from Zenodo.

## Notebooks

| Notebook | What it does | Open |
|---|---|---|
| `01_explore_data` | EDA: metadata vs. radar data, class balance, radar images, which variables/sweep/frame to use | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baevv/tornado-cnn/blob/main/notebooks/01_explore_data.ipynb) |
| `02_preprocess` | Preprocessing demo on one sample, all selected years saved to Google Drive, train/validation/test split with leakage checks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baevv/tornado-cnn/blob/main/notebooks/02_preprocess.ipynb) |
| `03_train_baseline` | Data loader, our CNN, training with early stopping, learning curves, validation metrics (GPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baevv/tornado-cnn/blob/main/notebooks/03_train_baseline.ipynb) |
| `04_hyperparameters` | One-at-a-time hyperparameter comparison (learning rate, schedule, L2, Dropout, width, batch size, optimizers SGD / momentum / RMSprop / Adam, pos_weight), per-category results (GPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baevv/tornado-cnn/blob/main/notebooks/04_hyperparameters.ipynb) |

## Project structure

```
notebooks/
  01_explore_data.ipynb   EDA: data structure, catalog, radar images, evidence for preprocessing choices
  02_preprocess.ipynb     preprocessing demo, process selected years to Drive, train/val/test split
  03_train_baseline.ipynb data loader, CNN, training, learning curves, validation results
  04_hyperparameters.ipynb hyperparameter comparison table
src/
  helper/
    paths.py              storage locations on Colab / locally
    tornet.py             download, catalog, reading and preprocessing samples
    eda.py                raw-data analysis: per-variable measures, rotation strength, separability (AUC)
    data.py               train/val/test split by storm episode and its checks
    model.py              the CNN
    train.py              data loading, DataLoader, loss weights, training loop, metrics
    viz.py                radar plots
  gui/
    episode_viewer.py     interactive storm episode (.npz + catalog.csv) viewer
configs/
  data.json               years, channels, sweep, frame
  train.json              training settings (optimizer, learning rate, dropout, ...)
results/                  figures and metric tables
```

## Running on Google Colab

1. Click an **Open in Colab** button above.
2. Run the first cell. It mounts Google Drive, clones this repo and installs missing packages.
   Allow Drive access when asked; processed data is stored there.
3. Run the remaining cells in order.

Preprocessing (`02_preprocess.ipynb`) needs no GPU; use a CPU runtime. Training will use a GPU runtime.

If the repository is private, add a GitHub token with read access as `GITHUB_TOKEN` in Colab Secrets
(key icon on the left, enable *Notebook access*). Never paste a token into a notebook or commit it.

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

- [x] 01 · Data exploration (EDA)
- [x] 02 · Preprocessing and train/validation/test split by storm episode
- [x] 03 · Data loader and CNN baseline with class-weighted loss
- [ ] 04 · Hyperparameter comparison (learning rate, schedule, L2, Dropout, width, batch size, optimizers SGD / momentum / RMSprop / Adam, pos_weight) (ready to run)
- [ ] 05 · Model comparison: our CNN vs. transfer learning (ResNet-18)
- [ ] 06 · Evaluation on the test split (accuracy, F1, ROC-AUC, PR-AUC, confusion matrix)
- [ ] 07 · Explainability (Grad-CAM)
- [ ] 08 · Interface to test the model on new samples
- [ ] Report (PDF)

## Citation and license

Data: TorNet, MIT Lincoln Laboratory, licensed under CC BY 4.0
([Zenodo](https://doi.org/10.5281/zenodo.12636522)).

> Veillette, M. S., et al. (2025). A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution
> Polarimetric Weather Radar Data. *Artificial Intelligence for the Earth Systems*, 4(1).
