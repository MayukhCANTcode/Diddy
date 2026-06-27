# Deforestation Detection using Deep Learning

An industry-level, research-oriented deep learning and computer vision framework for detecting forest cover changes and deforestation events using multispectral satellite telemetry.

This repository is designed to bridge the gap between experimental prototyping (Jupyter notebooks), production-grade engineering (modular library and CLI), and scientific publishing (LaTeX manuscripts).

---

## 📂 Repository Structure

Below is the directory tree of the repository:

```text
Deforestation-Detection/
├── .gitignore                  # Excludes large binaries (datasets, weights, latex temp files)
├── README.md                   # Project index, system architecture, and guide
├── requirements.txt            # Python packages (PyTorch, Rasterio, GeoPandas, etc.)
│
├── configs/                    # Experiment parameters (decoupled from code)
│   ├── data_config.yaml        # Bands, resolutions, paths, and patch parameters
│   └── model_config.yaml       # Hyperparameters, architectures, encoders, optimizer settings
│
├── data/                       # Dataset directories (ignored by git except placeholders)
│   ├── raw/                    # Unmodified original datasets
│   │   ├── eurosat/            # Sentinel-2 benchmark for land use classification
│   │   ├── sentinel2/          # Multi-temporal Sentinel-2 GeoTIFF tiles
│   │   └── global_forest_watch/# Vector/raster deforestation alerts and validation masks
│   ├── processed/              # Cropped 256x256 image patches and paired label masks
│   └── metadata/               # CSV lists indexing file splits and metadata labels
│
├── src/                        # Main modular library code
│   ├── __init__.py             # Exposes package metadata
│   ├── data/                   # Data pipeline loaders
│   │   ├── __init__.py
│   │   ├── dataset.py          # PyTorch Datasets for Sentinel-2, EuroSAT, and GFW
│   │   └── patch_gen.py        # Raster slicing utility using windowed cropping
│   ├── models/                 # Deep learning networks
│   │   ├── __init__.py
│   │   └── unet.py             # U-Net semantic segmentation network
│   ├── change_detection/       # Bi-temporal change processing
│   │   ├── __init__.py
│   │   └── detector.py         # Siamese or concatenation network for temporal difference maps
│   ├── pipelines/              # Orchestration entry points
│   │   ├── __init__.py
│   │   ├── train.py            # Train script entry point
│   │   └── predict.py          # Inference script entry point
│   └── utils/                  # Helper utilities
│       ├── __init__.py
│       ├── metrics.py          # Paper metrics (IoU, F1-Score, Confusion Matrix)
│       └── visualization.py    # Matplotlib RGB overlays and side-by-side plots
│
├── notebooks/                  # Interactive experimentation
│   ├── 01_data_exploration.ipynb    # Visualizes satellite bands & metadata
│   ├── 02_patch_generation_demo.ipynb# Prototyping tiling logic
│   └── 03_model_evaluation.ipynb    # Runs prediction overlays & inspects errors
│
├── models/                     # Checkpoints and serializations (ignored by git)
│   ├── checkpoints/            # Epoch-wise training checkpoints
│   └── final/                  # Final production weights ready for inference
│
├── outputs/                    # Output artifacts for publication
│   ├── figures/                # Matplotlib graphs, qualitative maps, and charts
│   └── predictions/            # Raw GeoTIFF inference outputs and change maps
│
├── paper/                      # LaTeX publication resources
│   ├── figures/                # Diagrams and qualitative maps used in the paper
│   ├── sections/               # Modular LaTeX section files
│   ├── main.tex                # LaTeX master manuscript (IEEE TGRS format)
│   └── references.bib          # BibTeX citation records
│
└── tests/                      # Unit testing framework
    ├── test_data.py            # Validates shape matching, transforms, and nodata filters
    ├── test_models.py          # Validates forward passes, channel alignments, and shapes
    └── test_utils.py           # Validates metric outputs (IoU, Precision, Recall)
```

---

## 📖 Component Explanations

### 1. Configuration (`configs/`)
- **`data_config.yaml`**: Outlines spatial patch constraints (e.g. `patch_size: 256`, `stride: 128`), Sentinel-2 channel selection (`B02`, `B03`, `B04`, `B08`), and random seed splits.
- **`model_config.yaml`**: Controls training variables (learning rate, optimizer, batch size) and network settings (backbone encoders, segmentation classes), making execution files general and flexible.

### 2. Isolated Datasets (`data/`)
Decoupling datasets into `raw/` and `processed/` is a core ML pipeline best practice:
- **`data/raw/`**: Holds unmodified satellite tiles and GFW shapefiles. Never modify files in this directory; they represent your gold-standard ground truth.
- **`data/processed/`**: Holds cropped, cleaned, and split image-label tensor pairs generated by the patch slicing code. Separating this speeds up training since reading small patches is significantly faster than querying massive GeoTIFF tiles in real-time.

### 3. Modular Library (`src/`)
Rather than placing script logic in notebooks, all core code is written as a modular, importable Python package:
- **`data/dataset.py` & `data/patch_gen.py`**: Handle remote sensing IO using `rasterio` and generate tensor arrays.
- **`models/unet.py`**: Model building blocks.
- **`change_detection/detector.py`**: Logic for comparing images between two dates.
- **`utils/metrics.py` & `utils/visualization.py`**: Shared computation of statistics and visual overlays.

### 4. Jupyter Notebooks (`notebooks/`)
Notebooks are restricted to exploratory analysis, exploratory visualization, and post-hoc error analysis. Any reusable code blocks (e.g., loading data or setting up models) are imported from `src/` to prevent copy-paste synchronization errors.

### 5. Academic Paper Drafts (`paper/`)
Houses LaTeX source code (`main.tex`, `references.bib`). Putting the writing pipeline in the codebase allows you to directly script data tables, loss plots, and prediction figures from the `outputs/` folder directly into the LaTeX document compilation flow.

---

## 📈 Scalability and Professional Suitability

This architecture is tailored to make your research portfolio-ready and publishable for several reasons:

1. **Clean Code & Git History (No Large Binaries)**
   By ignoring raw data (`data/raw/*`), processed tensors (`data/processed/*`), and training weights (`models/**/*.pth`), the repository remains lightweight (<10MB). Collaborators can clone it instantly, download data separately, and run scripts without bloating Git storage.
   
2. **Notebook-to-Script Transition**
   By putting the core codebase in `src/`, notebooks remain simple visualization wrappers. This shows employers and reviewers that you write clean, production-ready, testable python files rather than relying solely on monolithic, un-testable notebook files.
   
3. **Decoupled Configuration**
   Separating hyperparameters and filesystem paths from the python code ensures you can run different experiments (e.g. U-Net vs DeepLabv3+, or RGB vs RGB+NIR bands) simply by passing different configuration files, without modifying a single line of training code.
   
4. **Co-located Code and Publication Source**
   Co-locating your paper and source code means that figures generated inside `outputs/figures/` can be symlinked or referenced directly by `paper/main.tex`. When you update your model and regenerate plots, your LaTeX paper automatically updates with the latest results.
