# Training a Playlist Curator Based on User Taste

A supervised multi-class classifier that learns what kind of songs belong in each
of a listener's playlists, then predicts the most suitable playlist for a new song.

**Input (per song):** `danceability, energy, key, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo` + genre tags
**Output:** the best-matching playlist (with a probability for every playlist)

Three models are trained and compared:

| Model | Library |
|-------|---------|
| Logistic Regression | scikit-learn |
| SVM with RBF kernel | scikit-learn |
| Feed-forward neural network (one hidden layer of 64 units) | PyTorch |

## Project structure

```
playlist-curator/
├── README.md
├── requirements.txt
├── .gitignore
├── app/
│   └── app.py                # Streamlit web app
├── data/
│   ├── raw/                  # put the 7 *_music_data.csv files here
│   └── processed/            # train.csv / test.csv (generated)
├── models/                   # saved models + preprocessor (generated)
├── reports/
│   └── figures/              # plots (generated)
└── src/
    ├── __init__.py
    ├── config.py             # paths, columns, playlists, hyper-parameters
    ├── data_loader.py        # read raw CSVs and processed splits
    ├── features.py           # genre encoding, key one-hot, scaling
    ├── preprocess.py         # step 1: clean, de-duplicate, split, fit encoders
    ├── models.py             # LogReg, SVM, PyTorch network, save/load
    ├── train.py              # step 2: train and save all models
    ├── evaluate.py           # step 3: metrics, reports, confusion matrices
    ├── visualize.py          # step 4: PCA plots
    ├── tune.py               # optional: 5-fold CV + C / gamma search
    ├── predict.py            # PlaylistCurator class + command-line demo
    └── utils.py              # console helper
```

## Setup

You need Python 3.9 or newer (3.10 - 3.12 recommended). Run every command from the
project root (the folder that contains `requirements.txt`).

**Windows (PowerShell):**
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then unzip the dataset and copy the seven CSV files
(`alternative_music_data.csv`, `blues_music_data.csv`, `hiphop_music_data.csv`,
`indie_alt_music_data.csv`, `metal_music_data.csv`, `pop_music_data.csv`,
`rock_music_data.csv`) into `data/raw/`.

## How to run

```bash
python -m src.preprocess     # 1. clean data, split, fit encoders
python -m src.train          # 2. train + save the three models
python -m src.evaluate       # 3. test-set metrics, reports, confusion matrices
python -m src.visualize      # 4. PCA plots
python -m src.predict        # 5. (optional) demo predictions on random test songs
streamlit run app/app.py     # 6. launch the web app
```

Optional: `python -m src.tune` runs 5-fold cross-validation on the training set and
searches for a better `C` (and `gamma`). Copy the best values into `src/config.py`,
then re-run `train` and `evaluate`.

Outputs appear in `data/processed/`, `models/`, `reports/` and `reports/figures/`.

## How it works

**The data.** The seven CSVs contain 347 overlapping Spotify playlists. Learning
all 347 from audio features is unrealistic, so the project models *one listener's
taste* as 10 clearly different playlists, listed in `SELECTED_PLAYLISTS` in
`src/config.py`. To use different playlists, run
`python -m src.preprocess --list-playlists`, edit that list, and repeat steps 1-4.

**Cleaning (`src/preprocess.py`).**
- Tracks with a missing value or `tempo == 0` (failed audio analysis) are dropped.
- A track that appears twice in the same playlist is kept once.
- A track that appears in two *different* selected playlists is removed, because a
  single-label classifier cannot know which label is "right".
- Every remaining track therefore appears exactly once, so no song can be in both
  the training and the test set.

**Split.** 80 % train / 20 % test, stratified by playlist (fixed random seed).
The neural network also holds out 10 % of the *training* set for early stopping;
the test set is only used in `evaluate.py`.

**Features (`src/features.py`).** All preprocessing is fitted on the training set
only and saved as `models/preprocessor.joblib`:

| Feature | Treatment |
|---------|-----------|
| danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo | `StandardScaler` |
| mode | kept as 0 / 1 |
| key | one-hot (12 columns), since key 11 is not "bigger" than key 0 |
| genre tags | multi-hot over the 50 most frequent tags in the training set |

That gives 72 input features. Songs with no genre tags get all-zero genre columns.
Scaling matters a lot for the SVM and linear model (the reference paper's scaled vs
unscaled results differ enormously), which is why it is part of the saved preprocessor.

**Models.** Logistic Regression and the RBF SVM use `class_weight="balanced"`;
the neural network uses the equivalent class-weighted loss. All three output
probabilities, and the predicted playlist is always the one with the highest
probability, so evaluation matches exactly what the app shows.

**Evaluation.** Accuracy, macro-F1, weighted-F1, top-3 accuracy (is the true
playlist among the 3 best guesses?), per-playlist precision/recall, and
row-normalised confusion matrices.

## Generated files

| File | Created by |
|------|-----------|
| `data/processed/train.csv`, `test.csv` | `preprocess` |
| `models/preprocessor.joblib`, `label_encoder.joblib` | `preprocess` |
| `models/logistic_regression.joblib`, `svm_rbf.joblib`, `neural_net.pt` | `train` |
| `reports/figures/nn_training_curves.png` | `train` |
| `reports/cv_results.csv` | `tune` (optional) |
| `reports/model_comparison.csv` | `evaluate` |
| `reports/classification_report_<model>.txt` | `evaluate` |
| `reports/figures/confusion_matrix_<model>.png`, `model_comparison.png` | `evaluate` |
| `reports/figures/pca_playlists.png`, `pca_explained_variance.png` | `visualize` |

## Using the model from Python

```python
from src.predict import PlaylistCurator

curator = PlaylistCurator()
song = {
    "danceability": 0.45, "energy": 0.91, "key": 0, "loudness": -9.5, "mode": 1,
    "speechiness": 0.03, "acousticness": 0.002, "instrumentalness": 0.0,
    "liveness": 0.13, "valence": 0.75, "tempo": 116.0,
    "genres": ["hard rock", "rock"],
}
print(curator.predict(song, model_name="svm_rbf", top_k=3))
```

Valid `model_name` values: `logistic_regression`, `svm_rbf`, `neural_net`.

## Relation to the reference paper

This project follows *Training a Playlist Curator Based on User Taste*
(Awadelkarim & Coelho, 2018): audio features plus genre tags, a per-user
multi-class playlist classifier, an 80/20 split, scaled features, comparison of
a linear model, an RBF SVM and a small neural network, and PCA plots of
playlist separability.

Deliberate differences:
- **Logistic Regression** replaces the paper's perceptron (both are linear
  one-vs-rest style classifiers; the paper notes the perceptron rule generalises
  logistic regression's gradient update).
- **`instrumentalness`** is included as an 11th audio feature.
- **Genres** use the 50 most common tags instead of all ~2000 (`GENRE_TOP_N` in
  `src/config.py`; set it to `None` to use every tag in the training set).
- **Neural network** trains with Adam on mini-batches, class-weighted loss and
  early stopping, instead of full-batch gradient descent for a fixed 150 epochs.
- **Tracks that appear in several selected playlists are removed.** The paper
  keeps them, which makes some labels contradictory.
- The paper found classification gets easier with fewer playlists ("segmentation"):
  to try this, shorten `SELECTED_PLAYLISTS` to 3-5 entries.

## Troubleshooting

- **`No files matching '*_music_data.csv'`** - copy the CSVs into `data/raw/`.
- **`... not found. Run python -m src.preprocess first`** - run the steps in order.
- **`ModuleNotFoundError: No module named 'src'`** - run commands from the project
  root, and use `python -m src.xxx` (not `python src/xxx.py`).
- **Several Python installs (Windows)** - activate the virtual environment first;
  `python` then always points to the environment's own interpreter. Check with
  `python -c "import sys; print(sys.executable)"`.
- **Re-running `preprocess`** invalidates the saved models, so run `train` again afterwards.

## Ideas for extensions

Tune `C` / `gamma` with `GridSearchCV`, try different `SELECTED_PLAYLISTS`, change
`GENRE_TOP_N`, or add `popularity` / `duration_ms` as extra features.
