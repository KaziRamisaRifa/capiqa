import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Blocks
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class CPA(nn.Module):
    """ Cross Prompt Attention with DyT 'norm' and concat residual (2*dim). """
    def __init__(self, text_dim=768, attn_heads=4, alpha_init=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=attn_heads, batch_first=True)
        self.norm1 = DyT(2*text_dim, alpha_init_value=alpha_init)
        self.ffn = nn.Sequential(
            nn.Linear(2*text_dim, 2*text_dim),
            nn.ReLU(),
            nn.Linear(2*text_dim, 2*text_dim)
        )
        self.norm2 = DyT(2*text_dim, alpha_init_value=alpha_init)

    def forward(self, q, k, v):
        attn_out, _ = self.attn(q, k, v)         # [B, 1, d]
        x = torch.cat([attn_out, q], dim=-1)     # [B, 1, 2d]
        x2 = self.ffn(self.norm1(x))
        out = self.norm2(x + x2)                 # [B, 1, 2d]
        return out

class ContextBranch(nn.Module):
    """
    MLP on pooled image feature -> text_dim; adds L learnable context tokens.
    """
    def __init__(self, feat_dim, text_dim=768, L=4):
        super().__init__()
        self.L = L
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, text_dim)
        )
        self.context_tokens = nn.Parameter(torch.randn(L, text_dim))  # [L, text_dim]

    def forward(self, feat):
        # feat: [B, feat_dim]
        h_feat = self.mlp(feat)                                     # [B, text_dim]
        context = self.context_tokens.unsqueeze(0) + h_feat.unsqueeze(1)  # [B, L, text_dim]
        return context

# ----------------------------
# Main model: UNet_IQA (SUM only)
# ----------------------------
class UNet_IQA(nn.Module):
    def __init__(self, in_ch=1, base_dim=32, text_dim=768, context_L=4):
        super().__init__()
        features = [base_dim, base_dim*2, base_dim*4, base_dim*8]

        # Encoder
        self.downs = nn.ModuleList()
        chs = in_ch
        for feat in features:
            self.downs.append(DoubleConv(chs, feat))
            chs = feat
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.feat_dim = features[-1]*2  # e.g., 512 when base_dim=32

        # Text/prompt interaction
        self.q_proj = nn.Linear(self.feat_dim, text_dim)
        self.context_branch = ContextBranch(feat_dim=self.feat_dim, text_dim=text_dim, L=context_L)
        self.cpa = CPA(text_dim=text_dim)
        self.cpa_to_feat = nn.Linear(2*text_dim, self.feat_dim)  # projects CPA output back to feat_dim

        # (Optional) Decoder pieces are kept but unused for score-only path
        self.ups = nn.ModuleList()
        chs = self.feat_dim
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(chs, feat, 2, stride=2))
            self.ups.append(DoubleConv(chs, feat))
            chs = feat
        self.final = nn.Conv2d(features[0], 1, 1)

        # Regression head → [0,4] range
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B, feat_dim, 1, 1]
            nn.Flatten(),                # [B, feat_dim]
            nn.Linear(self.feat_dim, 1), # -> [B, 1]
            nn.Sigmoid()                 # -> [0,1], scaled to [0,4] below
        )

    def encode(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        return x, skips

    def decode(self, x, skips):
        # kept for future use if you want full UNet output
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)
        return self.final(x)

    def forward(self, img, prompt_emb):
        """
        img:        [B, 1, H, W]
        prompt_emb: [B, 1, 768]  (single prompt token per sample)
        Returns:
            score:   [B] in [0,4]
        """
        encoded, skips = self.encode(img)                # [B, C, H', W']
        gap = encoded.mean(dim=[2, 3])                   # [B, C]

        # Build prompt tokens = learned context (L) + provided prompt (1)
        context_vecs = self.context_branch(gap)          # [B, L, 768]
        prompt_tokens = torch.cat([context_vecs, prompt_emb], dim=1)  # [B, L+1, 768]

        # Q from image features, K/V from prompt tokens
        q_proj = self.q_proj(gap).unsqueeze(1)           # [B, 1, 768]
        Q = F.layer_norm(q_proj, q_proj.shape[-1:])      # [B, 1, 768]
        K = V = F.layer_norm(prompt_tokens, prompt_tokens.shape[-1:])  # [B, L+1, 768]

        # Cross Prompt Attention -> fuse into image features
        cpa_out = self.cpa(Q, K, V)                      # [B, 1, 2*768]
        cpa_proj = self.cpa_to_feat(cpa_out)             # [B, 1, C]
        cpa_feat = (
            cpa_proj.squeeze(1).unsqueeze(-1).unsqueeze(-1)
            .expand(-1, self.feat_dim, encoded.shape[2], encoded.shape[3])
        )                                                # [B, C, H', W']

        fused = encoded + cpa_feat

        # Score (no decoder needed)
        score = self.reg_head(fused) * 4.0               # [B, 1] -> [0,4]
        return score.squeeze(1)                          # [B]
