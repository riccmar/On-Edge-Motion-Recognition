## Introduction

This project implements a complete **Human Activity Recognition (HAR)** pipeline that runs
on the **BrainChip Akida 1000** neuromorphic processor. HAR from wearable sensors is a key
building block of Artificial Intelligence of Things (AIoT) applications (healthcare, fitness,
ambient assisted living), but continuous inference on battery-powered wearables makes standard,
power-hungry ANNs a poor fit.

Neuromorphic computing addresses this by using **Spiking Neural Networks (SNNs)**, which process
information as sparse, asynchronous binary events (spikes) and therefore consume energy only when
and where a spike is generated. We design a Convolutional Neural Network (CNN) that strictly
respects the Akida 1.0 hardware constraints, train it in floating point, apply
**Quantization-Aware Training (QAT)**, convert it to an SNN with `cnn2snn`, and deploy it on the
physical AKD1000 board.

We use the **WISDM** dataset (watch accelerometer + gyroscope, 20 Hz) and evaluate two scenarios:

- **Small (7 classes)**: a hand-oriented activity subset, used to benchmark against prior
  neuromorphic literature.
- **Big (18 classes)**: a broader subset used to stress-test model capacity, with the
  architecture and hyperparameters optimized via **Optuna**.

### Results (on physical Akida 1000)

| Scenario         | # Params | Memory (MB) | Keras Float32 | Post-QAT | Akida (SNN) | Throughput (FPS) |
|------------------|---------:|------------:|--------------:|---------:|------------:|-----------------:|
| Small (7 classes)|  138,653 |        0.54 |        94.55% |   91.18% |      91.41% |         2,012.03 |
| Big (18 classes) |  304,342 |        1.16 |        80.22% |   72.81% |      72.82% |         2,649.21 |

The 7-class result (91.41%) closely matches the ~92.5% baseline of Fra et al. [1] with a smaller
memory footprint, and shows **zero accuracy drop** between the software simulator and the physical
silicon. A detailed write-up is available in the accompanying project report/paper.

---

## Objective

Demonstrate that a complex HAR task can be deployed end-to-end on ultra-low-power neuromorphic
hardware while staying competitive with the literature. Concretely, the project:

1. Prepares and windows the WISDM watch data (2 s windows, 40 samples, 6 channels).
2. Designs a CNN that satisfies the Akida 1.0 constraints:
   - symmetric padding only (input pre-padded to odd dimensions `41 x 9`),
   - no pooling (dimensionality reduction via strided convolutions),
   - bounded activations (`ReLU6` instead of `ReLU`),
   - no Batch Normalization (avoids silent fusion errors at conversion),
   - a `1 x 1` pointwise convolution before `Flatten` to keep the model within on-chip SRAM.
3. Runs the three-phase deployment pipeline: **CNN training &rarr; Quantization and QAT (8/4/4 first layer,
   4/4/4 hidden layers) &rarr; SNN conversion (`.fb`) and hardware mapping**.
4. Uses **Optuna** to search architecture/hyperparameters (Big model) and QAT hyperparameters.

---

## Repository structure

```
.
├── src/                              # Data prep + training / quantization / conversion
│   ├── 01_Dataset_Exploration.ipynb  # Download WISDM, fuse sensors, window, normalize, save .npy
│   ├── 02_SmallModel_train.ipynb     # 7-class: manual CNN -> QAT -> Akida .fb
│   └── 03_BigModel_train_optuna.ipynb# 18-class: Optuna CNN + QAT -> Akida .fb
├── test/                             # Evaluation notebooks (CNN / quantized / Akida / hardware)
│   ├── Evaluation_SmallModel.ipynb
│   └── Evaluation_BigModel.ipynb
├── models/
│   ├── small/                        # bestModel.h5, *_quantized.h5, *_quantized_akida.fb
│   └── big/                          # same, plus best_params/*.json (Optuna results)
├── images/                           # Architectures, confusion matrices, logo
├── requirements.txt
└── README.md
```

The notebooks auto-detect the environment. In Colab, `BASE_DIR = './'` and everything lives under
`/content/`; run locally, `BASE_DIR = '../'`, so paths resolve relative to the repo root when a
notebook is executed from inside `src/` or `test/`. Raw data, preprocessed `.npy` files, and
pretrained model artifacts are downloaded on demand via `gdown` when not already present.

---

## Reproducing the experiments

> **Important (Apple Silicon / macOS users):** the BrainChip toolchain
> (`akida`, `cnn2snn`, `quantizeml`) is **not available for macOS**. You have two options:
> **(A) run everything on Google Colab** (recommended), or **(B) build a Linux/x86 conda
> environment** that satisfies `requirements.txt`. Any step that touches quantization, SNN
> conversion, or the physical board must run in one of these environments. Note also that
> physical-hardware execution requires access to an actual Akida AKD1000 device
> (e.g. BrainChip's cloud/dev board); without it you can still run the Keras and quantized
> evaluations and the Akida software simulator.

### Option A &mdash; Google Colab (recommended)

The notebooks are written for Colab. For each notebook:

1. Upload the notebook to Colab (or open it from the repo).
2. Run the first setup cell. When `COLAB` is detected it downloads `requirements.txt` and runs
   `pip install -r requirements.txt` automatically; no manual setup is needed.
3. Run the remaining cells top to bottom. Datasets and pretrained models are fetched via `gdown`
   on demand.

### Option B &mdash; Local conda environment (Linux / x86_64)

```sh
# Create and activate the environment (Python 3.10 recommended for the Akida toolchain)
conda create -n har-akida python=3.10 -y
conda activate har-akida

# Install pinned dependencies
pip install -r requirements.txt

# Register the kernel and launch Jupyter
python -m ipykernel install --user --name har-akida --display-name "har-akida"
jupyter lab   # or: jupyter notebook
```

Key pinned versions (see `requirements.txt` for the full list):
`akida==2.19.1`, `cnn2snn==2.19.1`, `quantizeml==1.2.3`, `tensorflow==2.19.1`,
`tf_keras==2.19.0`, `numpy==2.0.2`, `scikit-learn==1.6.1`, plus `optuna`, `gdown`, `pandas`,
`matplotlib`, `joblib`.

> The pipeline forces the **legacy Keras 2 backend** via `os.environ["TF_USE_LEGACY_KERAS"] = "1"`
> (set at the top of the notebooks) because `cnn2snn` is incompatible with Keras 3. Keep this in
> place.

When running locally, launch Jupyter from the repo root (or from inside `src/`/`test/`) so that the
`BASE_DIR = '../'` relative paths resolve correctly.

### Suggested run order

1. **`src/01_Dataset_Exploration.ipynb`** &mdash; downloads WISDM, aligns accel+gyro, builds 2 s
   windows, applies z-score normalization, and saves the processed `.npy` datasets (full 18-class
   "big" set and the 7-class hand-oriented "small" subset). *Optional*: the training/evaluation
   notebooks can also download the preprocessed `.npy` files directly, so you can skip this step.
2. **`src/02_SmallModel_train.ipynb`** &mdash; trains the manually engineered 7-class CNN, runs
   QAT, converts to an Akida `.fb`, and saves artifacts under `models/small/`.
3. **`src/03_BigModel_train_optuna.ipynb`** &mdash; runs Optuna for the 18-class architecture and
   QAT hyperparameters (saved to `models/big/best_params/`), trains, quantizes, converts, and saves
   artifacts under `models/big/`.
4. **`test/Evaluation_SmallModel.ipynb`** / **`test/Evaluation_BigModel.ipynb`** &mdash; evaluate
   each stage (float CNN, quantized model, Akida SNN in the simulator, and, if a device is present,
   on the physical AKD1000) and produce the confusion matrices.

Reproducibility note: the train/val/test split is stratified with `random_state=42`
(70% / 15% / 15%).

---

## Authors

- Giosuè Pinto (s342711)
- Francesco Palmisani (s343429)
- Riccardo Marconi (s342227)

## References

[1] V. Fra, E. Forno, R. Pignari, T. C. Stewart, E. Macii, and G. Urgese,
"Human activity recognition: suitability of a neuromorphic approach for on-edge AIoT
applications," *Neuromorphic Computing and Engineering*, vol. 2, no. 1, p. 014006, 2022.
[https://doi.org/10.1088/2634-4386/ac4c38](https://doi.org/10.1088/2634-4386/ac4c38)
