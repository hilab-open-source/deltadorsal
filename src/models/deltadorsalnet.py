"""Core network architecture for DeltaDorsal."""

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV3_LOCATION = "submodules/dinov3"

MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

MODEL_TO_NUM_FEATURES = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1024,
    "dinov3_vit7b16": 1024,
}


class DINOFeaturizer(nn.Module):
    def __init__(self, dinov3_location, model_name, weights_path, n_unfrozen_blocks=3):
        super().__init__()
        if model_name not in MODEL_TO_NUM_LAYERS:
            raise ValueError(f"Invalid Dino Model Name: {model_name}")

        self.model = torch.hub.load(
            dinov3_location, model_name, source="local", weights=weights_path
        )
        self.n_layers = MODEL_TO_NUM_LAYERS[model_name]
        self.feature_dim = MODEL_TO_NUM_FEATURES[model_name]

        for p in self.model.parameters():
            p.requires_grad = False

        if n_unfrozen_blocks != 0:
            for block in self.model.blocks[-n_unfrozen_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

            for p in self.model.norm.parameters():
                p.requires_grad = True

    def forward(self, img):
        feats = self.model.get_intermediate_layers(
            img, n=range(self.n_layers), reshape=True, norm=True
        )
        return feats[-1]


class ChangeEncoder(nn.Module):
    def __init__(self, in_dim, mid_dim=256, use_delta=True, use_ft_f0=False):
        super().__init__()
        self.use_delta = use_delta
        self.use_ft_f0 = use_ft_f0

        # disable using the base entirely if we're not using deltas
        if not self.use_delta:
            self.use_ft_f0 = False

        base_feat_dim = in_dim * 2 if self.use_ft_f0 else 0

        delta_dim = 1 if self.use_delta else 0

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_dim + delta_dim + base_feat_dim, mid_dim, kernel_size=3, padding=1
            ),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, Ft, F0):
        if self.use_delta:
            dF = Ft - F0
            # cosine similarity per-location
            cos = F.cosine_similarity(Ft, F0, dim=1, eps=1e-6).unsqueeze(
                1
            )  # (B,1,H',W')
            x = [dF, cos]
            if self.use_ft_f0:
                x.extend([Ft, F0])
        else:
            x = [Ft]

        x = torch.cat(x, dim=1)
        return self.conv(x)


class ResidualPoseHead(nn.Module):
    def __init__(
        self,
        z_dim,
        out_dim,
        prior_dim=45,
        hidden_dim=512,
        use_film=False,
        use_prior=False,
        gated=False,
        pool="avg",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.use_film = use_film and use_prior
        self.use_prior = use_prior
        self.gated = gated
        self.pool = pool

        self.out_dim = out_dim
        self.prior_dim = prior_dim

        self.hidden_dim = hidden_dim

        if use_film:
            self.film = nn.Sequential(
                nn.Linear(prior_dim, self.z_dim * 2),
            )

        # simple neck before pooling
        self.neck = nn.Sequential(
            nn.Conv2d(self.z_dim, self.z_dim, 3, padding=1), nn.GELU()
        )

        if self.use_prior:
            self.mlp = nn.Sequential(
                nn.Linear(self.z_dim + prior_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim * 2 if self.gated else out_dim),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            with torch.no_grad():
                if self.gated:
                    self.mlp[-1].bias[: self.out_dim].zero_()
                    self.mlp[-1].bias[self.out_dim :].fill_(-2.0)  # small gates
                else:
                    self.mlp[-1].bias.zero_()
        else:
            self.mlp = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim * 2 if self.gated else out_dim),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            with torch.no_grad():
                if self.gated:
                    self.mlp[-1].bias[: self.out_dim].zero_()
                    self.mlp[-1].bias[self.out_dim :].fill_(2.0)
                else:
                    self.mlp[-1].bias.zero_()

    def forward(self, X, prior):
        B, C, H, W = X.shape

        # FiLM modulation
        if self.use_film:
            gb = self.film(prior)
            gamma, beta = gb[:, :C], gb[:, C:]
            gamma = gamma.view(B, C, 1, 1)
            beta = beta.view(B, C, 1, 1)
            X = gamma * X + beta

        X = self.neck(X)
        if self.pool == "avg":
            g = X.mean(dim=[2, 3])
        elif self.pool == "attn":
            # lightweight spatial attention pooling
            w = torch.tanh(nn.Conv2d(X.shape[1], 1, 1).to(X.device)(X))
            w = torch.softmax(w.view(X.size(0), -1), dim=-1).view(
                X.size(0), 1, X.size(2), X.size(3)
            )
            g = (X * w).sum(dim=[2, 3])
        else:
            g = X.mean(dim=[2, 3])

        if self.use_prior:
            z = torch.cat([g, prior], dim=1)
            out = self.mlp(z)
            if self.gated:
                dtheta, gate_logits = out.chunk(2, dim=1)
                gate = torch.sigmoid(gate_logits)
                theta_hat = prior + gate * dtheta
                return theta_hat, dtheta, gate
            else:
                dtheta = out
                theta_hat = prior + dtheta
                return theta_hat, dtheta, torch.ones_like(dtheta)
        else:
            z = g
            out = self.mlp(z)
            if self.gated:
                theta_raw, gates_logits = out.chunk(2, dim=1)
                gate = torch.sigmoid(gates_logits)
                theta_hat = gate * theta_raw
                return theta_hat, theta_raw, gate
            else:
                theta_hat = out
                return (
                    theta_hat,
                    theta_hat,
                    torch.ones(B, self.out_dim, device=X.device, dtype=X.dtype),
                )


class ForceHead(nn.Module):
    def __init__(self, z_dim, out_dim, n_classes=4, hidden_dim=512, pool="avg"):
        super().__init__()
        self.z_dim = z_dim

        self.pool = pool

        self.out_dim = out_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        # simple neck before pooling
        self.neck = nn.Sequential(
            nn.Conv2d(self.z_dim, self.z_dim, 3, padding=1), nn.GELU()
        )

        self.trunk = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.head = nn.Linear(hidden_dim, n_classes * out_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, X):
        B, C, H, W = X.shape
        X = self.neck(X)

        if self.pool == "avg":
            g = X.mean(dim=[2, 3])
        elif self.pool == "attn":
            # lightweight spatial attention pooling
            w = torch.tanh(nn.Conv2d(X.shape[1], 1, 1).to(X.device)(X))
            w = torch.softmax(w.view(X.size(0), -1), dim=-1).view(
                X.size(0), 1, X.size(2), X.size(3)
            )
            g = (X * w).sum(dim=[2, 3])
        else:
            g = X.mean(dim=[2, 3])

        h = self.trunk(g)

        logits = self.head(h).view(B, self.out_dim, self.n_classes)
        probs = F.softmax(logits, dim=-1)

        return logits, probs


class DeltaDorsalNet(nn.Module):
    def __init__(
        self,
        model_name,
        model_path,
        dinov3_location=DINOV3_LOCATION,
        out_dim=256,
        n_unfrozen_blocks=3,
        use_delta: bool = True,
        use_orig_feat: bool = False,
    ):
        super().__init__()
        self.backbone = DINOFeaturizer(
            dinov3_location, model_name, model_path, n_unfrozen_blocks=n_unfrozen_blocks
        )

        self.change = ChangeEncoder(
            self.backbone.feature_dim,
            mid_dim=out_dim,
            use_delta=use_delta,
            use_ft_f0=use_orig_feat,
        )

    def forward(self, I_t, I_0):
        Ft = self.backbone(I_t)
        F0 = self.backbone(I_0)
        X = self.change(Ft, F0)  # B, 256, H, W
        return X


if __name__ == "__main__":
    net = DeltaDorsalNet(
        "dinov3_vith16plus",
        "/home/whuang/code/wrinklesense/_DATA/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    )
