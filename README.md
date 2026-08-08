# Breast Cancer Wisconsin (Diagnostic) - Basic Analysis & Permutation Importance

Simple analysis of the [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) using only `numpy` and `matplotlib`.

## What it does
1. Loads the dataset directly from `wdbc.data`
2. Prints basic stats: shape, missing values, class balance, mean/min/max/std per feature
3. Plots class distribution and a feature histogram
4. Trains a simple **nearest-centroid classifier** (finds the "average" benign patient and "average" malignant patient, classifies by whichever is closer)
5. Runs **permutation importance**: shuffles one feature at a time and measures how much the accuracy drops, to see which features matter most

## Files
- `analysis.py` — the full script
- `wdbc.data` — the dataset (569 patients, 30 features)
- `wdbc.names` — official dataset description from UCI

## How to run
```bash
pip install numpy matplotlib
python analysis.py
```

This will print results to the console and save three plots:
- `class_counts.png`
- `mean_radius_histogram.png`
- `permutation_importance.png`

## Notes
- No pandas or scikit-learn required — pure numpy for data handling and modeling.
- The classifier is trained and evaluated on the same data for simplicity (not a proper train/test split), so accuracy numbers are optimistic. The goal here is to illustrate the *logic* of permutation importance, not to build a production-grade model.
