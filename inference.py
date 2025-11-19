import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import dataloader
from text_encoder_util import IQAPromptEncoder
from tqdm import tqdm
from model import UNet_IQA
from scipy.stats import pearsonr, spearmanr, kendalltau
import csv


# ---------------------------
# Dataset (same as train)
# ---------------------------
class IQADataset(Dataset):
    def __init__(self, data):
        """
        data: list of (img_np, score_float, fname)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, score, fname = self.data[idx]
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = torch.from_numpy(img).unsqueeze(0)  # [1,H,W]
        score = torch.tensor(score, dtype=torch.float32)
        return img, score, fname


# ---------------------------
# Prompt utilities
# ---------------------------
@torch.no_grad()
def build_mean_prompt(prompt_encoder, device, bins=(0, 1, 2, 3, 4)):
    """Average all discrete prompt bins → neutral prompt embedding."""
    embs = [prompt_encoder.encode_prompt(int(b)).detach() for b in bins]
    mean_prompt = torch.mean(torch.stack(embs, dim=0), dim=0)
    return mean_prompt.to(device)  # [1, dim]


@torch.no_grad()
def prompts_from_indices(prompt_encoder, idx, device):
    """
    Convert integer indices (shape [B]) to prompt embeddings [B,1,dim].
    """
    embs = [prompt_encoder.encode_prompt(int(i.item())).detach() for i in idx]
    prompt_embs = torch.cat(embs, dim=0).unsqueeze(1).to(device)
    return prompt_embs  # [B,1,dim]


# ---------------------------
# Inference / Evaluation
# ---------------------------
@torch.no_grad()
def run_inference(model, prompt_encoder, loader, device, mean_prompt):
    model.eval()

    preds, gts, fnames = [], [], []

    for img, score, fname in tqdm(loader, desc="Inference", dynamic_ncols=True):
        img = img.to(device)
        score = score.to(device)
        B = img.size(0)

        # Pass 1: neutral prompt → provisional prediction
        neutral = mean_prompt.unsqueeze(0).expand(B, 1, -1)
        pred0 = model(img, neutral)  # [B]

        # Map provisional prediction to discrete bins
        idx = torch.clamp(torch.round(pred0).long(), 0, 4)

        # Pass 2: predicted-bin prompts → final prediction
        prompt_embs = prompts_from_indices(prompt_encoder, idx, device)
        pred = model(img, prompt_embs)  # [B]

        preds.append(pred.cpu())
        gts.append(score.cpu())
        fnames.extend(list(fname))

    preds = torch.cat(preds).numpy().flatten()
    gts = torch.cat(gts).numpy().flatten()

    # Metrics
    try:
        plcc = pearsonr(preds, gts)[0]
    except Exception:
        plcc = np.nan
    try:
        srcc = spearmanr(preds, gts)[0]
    except Exception:
        srcc = np.nan
    try:
        krocc = kendalltau(preds, gts)[0]
    except Exception:
        krocc = np.nan

    rmse = float(np.sqrt(np.mean((preds - gts) ** 2))) if len(preds) > 0 else np.nan

    return preds, gts, fnames, plcc, srcc, krocc, rmse


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------ Load test data (LDCTIQA) ------------
    test_full = dataloader.load_data(args.img_dir, dataloader.read_scores(args.json))
    print(f"Total test images: {len(test_full)}")
    test_loader = DataLoader(IQADataset(test_full), batch_size=args.batch, shuffle=False)

    # ------------ Model ------------
    model = UNet_IQA(in_ch=1).to(device)

    print(f"Loading checkpoint from {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # ------------ Prompt encoder ------------
    prompt_encoder = IQAPromptEncoder('iqa_prompt_gemini.json')
    mean_prompt = build_mean_prompt(prompt_encoder, device)

    # ------------ Run inference ------------
    preds, gts, fnames, plcc, srcc, krocc, rmse = run_inference(
        model, prompt_encoder, test_loader, device, mean_prompt
    )

    print("\n=== Test Results ===")
    print(f"PLCC : {plcc:.4f}" if not np.isnan(plcc) else "PLCC : NaN")
    print(f"SRCC : {srcc:.4f}" if not np.isnan(srcc) else "SRCC : NaN")
    print(f"KROCC: {krocc:.4f}" if not np.isnan(krocc) else "KROCC: NaN")
    print(f"RMSE : {rmse:.4f}" if not np.isnan(rmse) else "RMSE : NaN")

    # ------------ Optional CSV save ------------
    if args.out_csv is not None:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "gt_score", "pred_score"])
            for fname, gt, pred in zip(fnames, gts, preds):
                writer.writerow([fname, float(gt), float(pred)])
        print(f"Saved per-image predictions to {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str, required=True,
                        help="LDCTIQA image directory (same format as training)")
    parser.add_argument('--json', type=str, required=True,
                        help="JSON with scores for LDCTIQA test set")
    parser.add_argument('--ckpt', type=str, required=True,
                        help="Path to trained iqa_sum_*.pth checkpoint")
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--out_csv', type=str, default=None,
                        help="Optional path to save predictions as CSV")

    args = parser.parse_args()
    main(args)
