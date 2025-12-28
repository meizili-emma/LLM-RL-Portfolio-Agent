import math
from typing import Dict, Literal, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---------- width policy ----------
def pick_hidden(
    in_f: int, out_f: int,
    policy: Literal["geomean","bottleneck","no_bottleneck"] = "geomean",
    alpha: float = 1.0, beta: float = 1.0,
    h_min: int = 8, h_max: int = 64
) -> int:
    if policy == "geomean":
        h = int(round(alpha * (in_f * out_f) ** 0.5))
    elif policy == "bottleneck":
        h = int(round(beta * min(in_f, out_f)))
    elif policy == "no_bottleneck":
        h = max(in_f, out_f)
    else:
        raise ValueError(f"unknown policy={policy}")
    return max(h_min, min(h, h_max))


# ---------- generic Linear stack ----------
class LinearStack(nn.Module):
    """
    depth = number of Linear layers (>=1)
    layout per layer: Linear -> [Dropout] -> [ReLU], with 'last_activation' toggle.
    """
    def __init__(
        self,
        in_f: int, out_f: int, depth: int = 1,
        hidden_f: Optional[int] = None,
        *,
        use_dropout_per_layer: bool = False,
        dropout_p: float = 0.05,
        activation: Optional[nn.Module] = None,
        last_activation: bool = True,
        hidden_policy: Literal["geomean","bottleneck","no_bottleneck"] = "geomean",
        alpha: float = 1.0, beta: float = 1.0, h_min: int = 8, h_max: int = 64
    ):
        super().__init__()
        assert depth >= 1
        act = activation if activation is not None else nn.ReLU()

        layers = []
        if depth == 1:
            layers.append(nn.Linear(in_f, out_f))
            if use_dropout_per_layer and dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            if last_activation:
                layers.append(act)
        else:
            h = hidden_f if hidden_f is not None else pick_hidden(
                in_f, out_f, policy=hidden_policy, alpha=alpha, beta=beta, h_min=h_min, h_max=h_max
            )
            layers.append(nn.Linear(in_f, h))
            if use_dropout_per_layer and dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            layers.append(act)

            for _ in range(depth - 2):
                layers.append(nn.Linear(h, h))
                if use_dropout_per_layer and dropout_p > 0:
                    layers.append(nn.Dropout(dropout_p))
                layers.append(act)

            layers.append(nn.Linear(h, out_f))
            if use_dropout_per_layer and dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            if last_activation:
                layers.append(act)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------- tiny self-attention over tickers ----------
class TinySelfAttention(nn.Module):
    """
    One transformer-style block over N tickers.
    Keep it very small for our dataset.
    - d_model: 32~48
    - n_heads: 2
    - ffn_depth: 1 or 2 (1 = single Linear; 2 = Linear-ReLU-Linear)
    """
    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        ffn_hidden: Optional[int] = None,
        ffn_depth: int = 1,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        use_layernorm: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.ln1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        hid = ffn_hidden if ffn_hidden is not None else max(2 * d_model, 32)
        if ffn_depth == 1:
            # single linear (lightest)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Dropout(ffn_dropout) if ffn_dropout > 0 else nn.Identity(),
            )
        else:
            # 2-layer FFN
            self.ffn = nn.Sequential(
                nn.Linear(d_model, hid),
                nn.ReLU(),
                nn.Dropout(ffn_dropout) if ffn_dropout > 0 else nn.Identity(),
                nn.Linear(hid, d_model),
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, d_model)
        x = self.ln1(tokens)
        out, _ = self.attn(x, x, x, need_weights=False)
        x = tokens + out                         # residual
        y = self.ln2(x)
        y = x + self.ffn(y)                      # residual
        return y                                 # (B, N, d_model)


# ---------- Hybrid feature extractor ----------
class HybridExtractor(nn.Module):
    """
    Observation Dict:
      - portfolio_state: (B, 1+N)
      - global_state   : (B, G)
      - trading_state  : (B, N, H)
      - valuation_state: (B, N, V)

    Per-ticker aggregation mode:
      - per_ticker_mode = "pool" | "flatten" | "attention"
        * pool: mean/max over N (loses identity, smallest)
        * flatten: concat N tokens into a long vector (keeps identity)
        * attention: TinySelfAttention over N tokens, then flatten

    Output: (B, fuse_dim) vector for the policy.
    """
    def __init__(
        self,
        num_tickers: int,
        hist_len: int,
        v_dim: int,
        g_dim: int,
        port_dim: int,  # 1 + N
        # sizes
        z_port: int = 16, z_glob: int = 16, z_trad: int = 16,
        val_hidden: int = 32, z_val: int = 32,  # z_val kept small if you plan attention/flatten
        fuse_dim: int = 128,
        # depths
        port_depth: int = 2, glob_depth: int = 2,
        trad_depth: int = 1,            # used when trading_mode='mlp'
        val_pt_depth: int = 1,          # per-ticker encoder (valuation/trading merged later)
        # policies
        hidden_policy_port: str = "geomean",
        hidden_policy_glob: str = "geomean",
        hidden_policy_trad: str = "geomean",
        hidden_policy_val_pt: str = "geomean",
        # trading encoder kind
        trading_mode: Literal["mlp","conv1d"] = "mlp",
        conv_out_channels: int = 16, conv_kernel_size: int = 3, conv_dilation: int = 1,
        # per-ticker aggregation
        per_ticker_mode: Literal["pool","flatten","attention"] = "flatten",
        pool: Literal["mean","max"] = "mean",  # when per_ticker_mode="pool"
        # attention config (only used if per_ticker_mode="attention")
        attn_d_model: int = 32, attn_heads: int = 2,
        attn_ffn_hidden: Optional[int] = None, attn_ffn_depth: int = 1,
        attn_dropout: float = 0.0, attn_ffn_dropout: float = 0.0,
        # ticker embeddings
        use_ticker_embed: bool = True, ticker_emb_dim: int = 8,
        # broadcast global/portfolio into tokens
        use_broadcast_ctx: bool = True,  # project (port,glob) and add to each ticker token
        # regularization (extractor)
        use_layernorm: bool = False,
        use_dropout: bool = False, dropout_p: float = 0.05,
        # activations
        last_act_port: bool = True, last_act_glob: bool = True,
        last_act_trad: bool = False, last_act_val_pt: bool = False,
        # misc
        dtype=torch.float32, assert_shapes: bool = True
    ):
        super().__init__()
        self.N = int(num_tickers); self.H = int(hist_len)
        self.V = int(v_dim);       self.G = int(g_dim)
        self.P = int(port_dim)
        self._dtype = dtype; self._assert_shapes = assert_shapes

        assert per_ticker_mode in ("pool", "flatten", "attention")
        self.per_ticker_mode = per_ticker_mode
        assert pool in ("mean","max")
        self.pool = pool

        # --- portfolio/global encoders ---
        self.port_net = LinearStack(
            self.P, z_port, depth=port_depth,
            use_dropout_per_layer=use_dropout, dropout_p=dropout_p,
            last_activation=last_act_port, hidden_policy=hidden_policy_port
        )
        self.glob_net = LinearStack(
            self.G, z_glob, depth=glob_depth,
            use_dropout_per_layer=use_dropout, dropout_p=dropout_p,
            last_activation=last_act_glob, hidden_policy=hidden_policy_glob
        )
        self.ln_port = nn.LayerNorm(z_port) if use_layernorm else nn.Identity()
        self.ln_glob = nn.LayerNorm(z_glob) if use_layernorm else nn.Identity()

        # --- per-ticker trading encoder ---
        self.trading_mode = trading_mode
        if trading_mode == "mlp":
            self.trad_seq = LinearStack(
                self.H, z_trad, depth=trad_depth,
                use_dropout_per_layer=use_dropout, dropout_p=dropout_p,
                last_activation=last_act_trad, hidden_policy=hidden_policy_trad
            )
            self.trad_conv = None; self.trad_head = None; self.trad_proj = None
        else:
            pad = ((conv_kernel_size - 1) // 2) * conv_dilation  # same-length for odd kernels
            self.trad_conv = nn.Conv1d(1, conv_out_channels, conv_kernel_size, padding=pad, dilation=conv_dilation)
            self.trad_head = nn.AdaptiveAvgPool1d(1)
            self.trad_proj = LinearStack(
                conv_out_channels, z_trad, depth=1,
                use_dropout_per_layer=use_dropout, dropout_p=dropout_p,
                last_activation=False
            )
            self.trad_seq = None
        self.ln_trad = nn.LayerNorm(z_trad) if use_layernorm else nn.Identity()

        # --- per-ticker valuation encoder ---
        # Minimal: depth=1 → Linear(V→z_val) per ticker.
        self.val_pt = LinearStack(
            self.V, z_val, depth=val_pt_depth,
            use_dropout_per_layer=use_dropout, dropout_p=dropout_p,
            last_activation=last_act_val_pt, hidden_policy=hidden_policy_val_pt
        )
        self.ln_val = nn.LayerNorm(z_val) if use_layernorm else nn.Identity()

        # --- optional ticker ID embedding ---
        self.use_ticker_embed = use_ticker_embed
        if use_ticker_embed:
            self.tok_emb = nn.Embedding(self.N, ticker_emb_dim)
        else:
            self.tok_emb = None
            ticker_emb_dim = 0

        # --- build token dimension d_tok (per ticker) ---
        # token per ticker = concat( trading_emb[z_trad], valuation_emb[z_val], ticker_emb[ticker_emb_dim] )
        self.d_tok = z_trad + z_val + ticker_emb_dim

        # --- broadcast context (project port+glob and add to tokens) ---
        self.use_broadcast_ctx = use_broadcast_ctx
        if use_broadcast_ctx:
            self.ctx_proj = nn.Linear(z_port + z_glob, self.d_tok)
        else:
            self.ctx_proj = None

        # --- per-ticker aggregation to a fixed feature vector ---
        if per_ticker_mode == "pool":
            # pooled vector -> fuse with (z_port,z_glob)
            per_ticker_out_dim = self.d_tok
            fuse_in = (z_port + z_glob) + per_ticker_out_dim
            self.ticker_fuse = LinearStack(fuse_in, fuse_dim, depth=1, use_dropout_per_layer=False, last_activation=True)

        elif per_ticker_mode == "flatten":
            # flatten N*tok -> project down -> fuse
            self.flat_proj = LinearStack(self.N * self.d_tok, self.d_tok, depth=1, use_dropout_per_layer=False, last_activation=True)
            fuse_in = (z_port + z_glob) + self.d_tok
            self.ticker_fuse = LinearStack(fuse_in, fuse_dim, depth=1, use_dropout_per_layer=False, last_activation=True)

        else:  # attention
            self.attn_in  = nn.Linear(self.d_tok, attn_d_model)
            self.attn = TinySelfAttention(d_model=attn_d_model, n_heads=attn_heads,
                                  ffn_hidden=attn_ffn_hidden, ffn_depth=attn_ffn_depth,
                                  attn_dropout=attn_dropout, ffn_dropout=attn_ffn_dropout,
                                  use_layernorm=True)
            self.attn_out = nn.Linear(attn_d_model, self.d_tok)
            # after attention, flatten tokens and project to small vector
            self.attn_flat_proj = LinearStack(self.N * self.d_tok, self.d_tok, depth=1, use_dropout_per_layer=False, last_activation=True)
            fuse_in = (z_port + z_glob) + self.d_tok
            self.ticker_fuse = LinearStack(fuse_in, fuse_dim, depth=1, use_dropout_per_layer=False, last_activation=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1.0 / math.sqrt(max(1, fan_in))
                nn.init.uniform_(m.bias, -bound, bound)
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _poolN(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        return x.mean(dim=1) if self.pool == "mean" else x.amax(dim=1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        port = obs["portfolio_state"].to(self._dtype)     # (B, 1+N)
        glob = obs["global_state"].to(self._dtype)        # (B, G)
        ts   = obs["trading_state"].to(self._dtype)       # (B, N, H)
        vs   = obs["valuation_state"].to(self._dtype)     # (B, N, V)

        if self._assert_shapes:
            B = port.shape[0]
            assert port.shape == (B, self.P)
            assert glob.shape == (B, self.G)
            assert ts.shape[1:] == (self.N, self.H)
            assert vs.shape[1:] == (self.N, self.V)

        # context features
        z_port = self.ln_port(self.port_net(port))  # (B, z_port)
        z_glob = self.ln_glob(self.glob_net(glob))  # (B, z_glob)

        # trading per ticker
        B = ts.shape[0]
        if self.trading_mode == "mlp":
            te = self.trad_seq(ts.reshape(B*self.N, self.H)).reshape(B, self.N, -1)  # (B,N,z_trad)
        else:
            feat = F.relu(self.trad_conv(ts.reshape(B*self.N, 1, self.H)))
            feat = self.trad_head(feat).squeeze(-1)                                  # (B*N, C)
            te   = self.trad_proj(feat).reshape(B, self.N, -1)                       # (B,N,z_trad)
        te = self.ln_trad(te)

        # valuation per ticker
        ve = self.val_pt(vs.reshape(B*self.N, self.V)).reshape(B, self.N, -1)        # (B,N,z_val)
        ve = self.ln_val(ve)

        # token per ticker
        tokens = torch.cat([te, ve], dim=-1)                                         # (B,N,d_trad+z_val)
        if self.tok_emb is not None:
            # add learned ticker ID embeddings
            idx = torch.arange(self.N, device=tokens.device).unsqueeze(0).expand(B, self.N)
            t_emb = self.tok_emb(idx)                                                # (B,N,ticker_emb_dim)
            tokens = torch.cat([tokens, t_emb], dim=-1)                              # (B,N,d_tok)

        # broadcast context (optional)
        if self.ctx_proj is not None:
            ctx = torch.cat([z_port, z_glob], dim=1)                                 # (B, z_port+z_glob)
            ctx = self.ctx_proj(ctx).unsqueeze(1)                                    # (B,1,d_tok)
            tokens = tokens + ctx                                                    # inject context

        if self.per_ticker_mode == "pool":
            pooled = self._poolN(tokens)                                             # (B,d_tok)
            fused_in = torch.cat([z_port, z_glob, pooled], dim=1)                    # (B, z_port+z_glob+d_tok)
            out = self.ticker_fuse(fused_in)                                         # (B, fuse_dim)

        elif self.per_ticker_mode == "flatten":
            flat = tokens.reshape(B, self.N * tokens.shape[-1])                      # (B, N*d_tok)
            flat_small = self.flat_proj(flat)                                        # (B, d_tok)
            fused_in = torch.cat([z_port, z_glob, flat_small], dim=1)                # (B, z_port+z_glob+d_tok)
            out = self.ticker_fuse(fused_in)                                         # (B, fuse_dim)

        else:  # attention
            attn_tokens = self.attn_in(tokens)         # (B,N,attn_d_model)
            attn_out = self.attn(attn_tokens)       # (B,N,attn_d_model)
            attn_out= self.attn_out(attn_out)      # (B,N,d_tok)
            flat = attn_out.reshape(B, self.N * tokens.shape[-1])                    # (B, N*d_tok)
            flat_small = self.attn_flat_proj(flat)                                   # (B, d_tok)
            fused_in = torch.cat([z_port, z_glob, flat_small], dim=1)                # (B, z_port+z_glob+d_tok)
            out = self.ticker_fuse(fused_in)                                         # (B, fuse_dim)

        return out  # (B, fuse_dim)

    # ----- utilities -----
    def count_parameters(self, by_module: bool = True) -> Tuple[int, Dict[str, int]]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not by_module:
            return total, {}
        parts = {
            "port_net": self.port_net,
            "glob_net": self.glob_net,
            "trad_seq_or_conv": self.trad_seq if self.trad_seq is not None else self.trad_conv,
            "trad_head_or_proj": self.trad_proj if hasattr(self, "trad_proj") and self.trad_proj is not None else self.trad_head,
            "val_pt": self.val_pt,
            "tok_emb": self.tok_emb,
            "ctx_proj": self.ctx_proj,
            "flat_proj": getattr(self, "flat_proj", None),
            "attn": getattr(self, "attn", None),
            "attn_flat_proj": getattr(self, "attn_flat_proj", None),
            "ticker_fuse": self.ticker_fuse,
        }
        breakdown = {}
        for name, mod in parts.items():
            if mod is None: continue
            breakdown[name] = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        return total, breakdown


# ---------- SB3 adapter ----------

class PortfolioFeatureExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible adapter around HybridExtractor.

    Infers:
      - N, H, V, G, P from observation_space dict,
    and exposes a features_dim = fuse_dim vector to PPO/Actor-Critic policy.
    """

    def __init__(self, observation_space: gym.spaces.Dict, **kwargs: Any):
        port_shape = observation_space["portfolio_state"].shape   # (1+N,)
        trad_shape = observation_space["trading_state"].shape     # (N, H)
        val_shape = observation_space["valuation_state"].shape    # (N, V_eff)
        glob_shape = observation_space["global_state"].shape      # (G_eff,)

        num_tickers = trad_shape[0]
        hist_len = trad_shape[1]
        v_dim = val_shape[1]
        g_dim = glob_shape[0]
        port_dim = port_shape[0]

        fuse_dim = int(kwargs.pop("fuse_dim", 128))

        super().__init__(observation_space, features_dim=fuse_dim)

        self.extractor = HybridExtractor(
            num_tickers=num_tickers,
            hist_len=hist_len,
            v_dim=v_dim,
            g_dim=g_dim,
            port_dim=port_dim,
            fuse_dim=fuse_dim,
            **kwargs,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.extractor(observations)