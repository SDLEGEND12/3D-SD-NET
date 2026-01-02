# 3D SD-NET 🧠
Volumetric Organ Segmentation using 3D Vision Transformers

---

## Dataset

This project uses the **BraTS 2021 Brain Tumor Segmentation Dataset**.

🔗 Dataset Link:  
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

> ⚠️ The dataset is not included in this repository due to size constraints.

---

## Dataset Setup

1. Download and extract the BraTS 2021 dataset from the link above.
2. Place the dataset inside the `data/` directory as shown below:


data/
└── BraTS2021_Training_Data/
├── BraTS2021_00000/
├── BraTS2021_00001/
└── ...


---

## Project Pipeline

### 1. Create Train / Validation Splits

Generate CSV files defining training and validation splits:

```bash
python scripts/create_splits.py
```

This creates:

data/splits/
├── train.csv
└── val.csv

2. Build Preprocessed Cache

Preprocess volumetric data and store cached tensors for faster training:
```
python scripts/build_cache.py
```

Cached files are stored locally under:

cache/


Cache files are ignored by Git and not pushed to the repository.

3. Train the Model

Train the 3D SD-NET model using the cached data:
```
python scripts/train.py
```

During training:

A checkpoint is saved every 5 epochs as:
```
outputs/checkpoints/last_checkpoint.pt
```

The best-performing model is saved as:
```
outputs/checkpoints/best_model.pt
```
Inference
1. Standard Inference

Run inference on volumetric data using the trained model:
```
python inferencing/inference.py
```
2. 3D Rendering / Visualization

Visualize segmentation results in 3D:
```
python inferencing/3DRendering.py
```
Notes

Raw datasets, cache files, and checkpoints are excluded from version control.

This repository contains code only for clean reproducibility.

Designed for research and academic experimentation on 3D medical images.

Acknowledgements

BraTS 2021 Dataset

PyTorch

MONAI


---



