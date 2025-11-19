import os
import json
import random
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def read_scores(path):
    with open(path) as f:
        return json.load(f)

def find_image(image_dir, fname):
    path = os.path.join(image_dir, fname)
    if os.path.exists(path): return path
    base, _ = os.path.splitext(fname)
    for ext in ['.tif', '.tiff', '.png', '.jpg']:
        alt = os.path.join(image_dir, base + ext)
        if os.path.exists(alt): return alt
    return None

def load_data(image_dir, score_dict):
    data = []
    for fname, score in score_dict.items():
        p = find_image(image_dir, fname)
        if not p:
            print(f"Missing: {fname}")
            continue
        img = np.array(Image.open(p))
        if img.ndim == 3: img = img[..., 0]
        data.append((img, score, os.path.basename(p)))
    return data

def load_kadid10k_data(kadid_dir):
    csv_path = os.path.join(kadid_dir, "dmos.csv")
    images_dir = os.path.join(kadid_dir, "images")
    df = pd.read_csv(csv_path, sep=',')
    data = []
    for idx, row in df.iterrows():
        img_name = row['dist_img']
        dmos = float(row['dmos']) - 1.0   # [1,5] → [0,4]
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            if img.mode == 'RGB':
                img = img.convert('L')      # Always grayscale!
            img = img.resize((512, 512), Image.BILINEAR)   # <<--- ADD THIS LINE
            img = np.array(img)
            data.append((img, dmos, img_name))
        else:
            print(f"Missing: {img_path}")
    print(f"KADID10k: Loaded {len(data)} images")
    return data

# def stratified_split(data, ratio=0.9, n_bins=10, seed=42):
#     """
#     Stratified split based on scores.
#     Args:
#         data: list of (img, score, fname)
#         ratio: proportion for train
#         n_bins: number of bins for discretizing continuous scores
#     """
#     scores = np.array([d[1] for d in data])
    
#     # Bin continuous scores into categories for stratification
#     bins = np.linspace(scores.min(), scores.max(), n_bins)
#     y_binned = np.digitize(scores, bins, right=True)

#     train, val = train_test_split(
#         data,
#         test_size=1-ratio,
#         stratify=y_binned,
#         random_state=seed
#     )
#     return train, val


# def split(data, ratio=0.9):
#     random.shuffle(data)
#     i = int(len(data) * ratio)
#     return data[:i], data[i:]

def split(data, ratio=0.9, n_bins=10, seed=42):
    """
    Stratified split based on IQA scores to ensure equal distribution.
    Args:
        data: list of (img, score, fname)
        ratio: proportion for training set (default 0.9)
        n_bins: number of bins for discretizing continuous scores
        seed: random seed
    """
    scores = np.array([d[1] for d in data])  # Extract IQA scores
    # Bin scores into discrete categories for stratification
    bins = np.linspace(scores.min(), scores.max(), n_bins)
    y_binned = np.digitize(scores, bins, right=True)

    train, val = train_test_split(
        data,
        test_size=1 - ratio,
        stratify=y_binned,
        random_state=seed,
    )

    print(f"[split] Stratified: train={len(train)} val={len(val)} bins={n_bins}")
    return train, val

if __name__ == "__main__":
    random.seed(42)
    base = os.path.dirname(os.path.abspath(__file__))

    train_dir = os.path.join(base, "dataset/ldctiqac/train/image")
    train_json = os.path.join(base, "dataset/ldctiqac/train/train.json")
    test_dir  = os.path.join(base, "dataset/ldctiqac/test/images")
    test_json = os.path.join(base, "dataset/ldctiqac/test/test.json")

    train_full = load_data(train_dir, read_scores(train_json))
    train, val = split(train_full)
    print(f"Train: {len(train)}, Val: {len(val)}")

    test = load_data(test_dir, read_scores(test_json))
    print(f"Test: {len(test)}\nData Loaded!")
