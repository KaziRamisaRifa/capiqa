# dataloader.py
import os
import json
import random
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Optional precise TIFF reader
try:
    import tifffile
except Exception:
    tifffile = None

# ---------------------------
# Config: old and new WL
# (change these here as needed)
# ---------------------------
REWINDOW = True          # set False to bypass (debug)
W_OLD, L_OLD = 1400, -300  # dataset's existing WL (assumed)
W_NEW, L_NEW = 400, 50     # your desired WL for training

# ---------------------------
# WL helpers
# ---------------------------
def window_level(x, w=300, c=40, ymin=0.0, ymax=1.0):
    """Apply (w,c) to HU array x -> [ymin,ymax] float32."""
    x = x.astype(np.float32)
    y = np.zeros_like(x, dtype=np.float32)
    lo = c - 0.5 - (w - 1) / 2.0
    hi = c - 0.5 + (w - 1) / 2.0
    y[x <= lo] = ymin
    y[x >= hi] = ymax
    m = (x > lo) & (x < hi)
    y[m] = ((x[m] - (c - 0.5)) / (w - 1) + 0.5) * (ymax - ymin) + ymin
    return y

def inverse_window(y, w_old, c_old, ymin=0.0, ymax=1.0, hu_min=-1000, hu_max=3000):
    """
    Approx inverse mapping: y (already window-leveled with (w_old, c_old)) -> pseudo-HU.
    """
    y = y.astype(np.float32)
    if (ymax - ymin) != 1.0:
        y = (y - ymin) / max(1e-6, (ymax - ymin))
    else:
        y = np.clip(y, 0.0, 1.0)
    x = (y - 0.5) * (w_old - 1.0) + (c_old - 0.5)
    return np.clip(x, hu_min, hu_max).astype(np.float32)

# ---------------------------
# IO helpers
# ---------------------------
def read_scores(path):
    with open(path) as f:
        return json.load(f)

def find_image(image_dir, fname):
    path = os.path.join(image_dir, fname)
    if os.path.exists(path): return path
    base, _ = os.path.splitext(fname)
    for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']:
        alt = os.path.join(image_dir, base + ext)
        if os.path.exists(alt): return alt
    return None

def _read_gray_float01(path):
    """
    Read as grayscale float32 ~[0,1], preserving precision if TIFF.
    Assumes the file on disk is already windowed to (W_OLD, L_OLD).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"] and tifffile is not None:
        img = tifffile.imread(path)
        img = np.asarray(img)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)
            # If not clearly [0,1], do a robust clip to [0,1]
            if img.max() > 1.01 or img.min() < -0.01:
                p1, p2 = np.percentile(img, [1.0, 99.0])
                if p2 > p1:
                    img = np.clip((img - p1) / (p2 - p1), 0, 1).astype(np.float32)
                else:
                    img = np.clip(img, 0, 1).astype(np.float32)
        return img

    # Fallback: PIL
    im = Image.open(path)
    if im.mode == 'RGB':
        im = im.convert('L')
    elif im.mode != 'L':
        im = im.convert('L')
    arr = np.array(im, dtype=np.float32)
    if arr.max() > 1.01:
        arr = arr / 255.0
    return arr

# ---------------------------
# Public loaders (unchanged signatures)
# ---------------------------
def load_data(image_dir, score_dict):
    """
    Returns list of tuples (recon_img, score, basename).
    recon_img is float32 in [0,1], computed as:
       recon = WL_new( inverse_WL_old( input ) )
    If REWINDOW=False, returns input as-is (normalized to [0,1]).
    """
    data = []
    for fname, score in score_dict.items():
        p = find_image(image_dir, fname)
        if not p:
            print(f"Missing: {fname}")
            continue

        img01 = _read_gray_float01(p)   # ~[0,1], already WL'ed at (W_OLD, L_OLD)

        if REWINDOW:
            # inverse old → pseudo-HU → apply new
            hu = inverse_window(img01, w_old=W_OLD, c_old=L_OLD, ymin=0.0, ymax=1.0)
            recon = window_level(hu, w=W_NEW, c=L_NEW, ymin=0.0, ymax=1.0)
            img_out = recon.astype(np.float32)
        else:
            img_out = img01.astype(np.float32)

        if img_out.ndim == 3:
            img_out = img_out[..., 0]

        data.append((img_out, score, os.path.basename(p)))
    return data

def load_kadid10k_data(kadid_dir):
    csv_path = os.path.join(kadid_dir, "dmos.csv")
    images_dir = os.path.join(kadid_dir, "images")
    df = pd.read_csv(csv_path, sep=',')
    data = []
    for _, row in df.iterrows():
        img_name = row['dist_img']
        dmos = float(row['dmos']) - 1.0   # [1,5] → [0,4]
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            im = Image.open(img_path)
            if im.mode == 'RGB':
                im = im.convert('L')
            im = im.resize((512, 512), Image.BILINEAR)
            arr = np.array(im, dtype=np.float32)
            if arr.max() > 1.01:
                arr = arr / 255.0
            data.append((arr, dmos, img_name))
        else:
            print(f"Missing: {img_path}")
    print(f"KADID10k: Loaded {len(data)} images")
    return data

def split(data, ratio=0.9, n_bins=10, seed=42):
    scores = np.array([d[1] for d in data])
    bins = np.linspace(scores.min(), scores.max(), n_bins)
    y_binned = np.digitize(scores, bins, right=True)
    train, val = train_test_split(
        data, test_size=1 - ratio, stratify=y_binned, random_state=seed
    )
    print(f"[split] Stratified: train={len(train)} val={len(val)} bins={n_bins}")
    return train, val

# ---------------------------
# Quick self-test
# ---------------------------
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
