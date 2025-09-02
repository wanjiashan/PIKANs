import os, time, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def display_dataframe_to_user(title: str, df):
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))

# =========================
# Utils
# =========================
def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def moving_average(x: torch.Tensor, k: int = 5):
    if k <= 1: return x
    pad = k // 2
    xp = torch.nn.functional.pad(x.unsqueeze(0).unsqueeze(0), (pad, pad), mode='replicate').squeeze(0).squeeze(0)
    w = torch.ones(k, device=x.device) / k
    out = torch.nn.functional.conv1d(xp.unsqueeze(0).unsqueeze(0), w.view(1,1,-1), padding=0).squeeze(0).squeeze(0)
    return out

# =========================
# Data
# =========================
def load_series(xlsx_path, dt_min=7.0):
    df = pd.read_excel(xlsx_path)
    assert 'No.' in df.columns and 'Displacement (mm)' in df.columns, "Excel需包含列: No., Displacement (mm)"
    t = (df['No.'].astype(float) - 1.0) * dt_min
    S = df['Displacement (mm)'].astype(float)
    v = S.diff().iloc[1:] / dt_min
    t_v = t.iloc[1:]

    mask = v > 1e-6
    t_v = t_v[mask].reset_index(drop=True)
    v   = v[mask].reset_index(drop=True)

    t_v_np = t_v.values.astype(np.float32)
    v_np   = v.values.astype(np.float32)
    t_last = float(t_v_np.max()) if len(t_v_np) else 0.0
    return t_v_np, v_np, t_last

# =========================
# KAN 组件
# =========================
class RBF1D(nn.Module):
    def __init__(self, in_features: int, num_basis: int = 16):
        super().__init__()
        self.in_features = in_features
        self.num_basis = num_basis
        centers = torch.linspace(-1.0, 1.0, num_basis).view(1, num_basis).repeat(in_features, 1)
        self.centers = nn.Parameter(centers)
        self.log_sigmas = nn.Parameter(torch.full((in_features, num_basis), math.log(0.5)))
    def forward(self, x: torch.Tensor):
        B, D = x.shape
        assert D == self.in_features
        x_exp = x.unsqueeze(-1)                          # [B, D, 1]
        c = self.centers.unsqueeze(0)                    # [1, D, M]
        sigma = torch.exp(self.log_sigmas).unsqueeze(0)  # [1, D, M]
        z = (x_exp - c) / (sigma + 1e-6)
        return torch.exp(-0.5 * z * z)                  # [B, D, M]

class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_basis: int = 16, residual: bool = True, bias: bool = True):
        super().__init__()
        self.rbf = RBF1D(in_features, num_basis)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, num_basis) * (1.0 / math.sqrt(in_features * num_basis)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.skip = nn.Linear(in_features, out_features, bias=False) if residual else None
        if self.skip is not None:
            nn.init.kaiming_uniform_(self.skip.weight, a=math.sqrt(5))
        self.norm = nn.LayerNorm(out_features)
    def forward(self, x: torch.Tensor):
        phi = self.rbf(x)                                  # [B, in, M]
        y = torch.einsum('bim,oim->bo', phi, self.weight)  # [B, out]
        if self.bias is not None: y = y + self.bias
        if self.skip is not None: y = y + self.skip(x)
        return self.norm(y)

# =========================
# Model
# =========================
class PIKANModel(nn.Module):
    def __init__(self, hidden=64, t_min=0.0, t_rng=1.0, arch: str = "kan", kan_basis: int = 16,
                 margin_low_ratio: float = 0.1, margin_high_ratio: float = 0.1):
        super().__init__()
        self.raw_t_min = float(t_min)
        self.raw_t_rng = float(t_rng)
        self.margin_low = margin_low_ratio * t_rng
        self.margin_high = margin_high_ratio * t_rng
        self.t_min = self.raw_t_min - self.margin_low
        self.t_rng = self.raw_t_rng + self.margin_low + self.margin_high

        self.softplus = nn.Softplus()
        self.arch = arch.lower()
        self.in_dim = 2
        self.hidden = hidden

        self.input_gamma = nn.Parameter(torch.ones(self.in_dim))
        self.input_beta  = nn.Parameter(torch.zeros(self.in_dim))
        self.bias_u = nn.Parameter(torch.tensor(0.0))   # ← 全局可学习偏置

        if self.arch == "mlp":
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 2)
            )
        elif self.arch == "kan":
            self.net = nn.Sequential(
                KANLayer(self.in_dim, hidden, num_basis=kan_basis, residual=True),
                nn.GELU(),
                KANLayer(hidden, hidden, num_basis=kan_basis, residual=True),
                nn.GELU(),
                KANLayer(hidden, 2, num_basis=kan_basis, residual=False),
            )
        else:
            raise ValueError(f"Unknown arch: {arch}")

    def forward(self, t_in, vhat_in):
        x = torch.cat([t_in, vhat_in], dim=1)           # [B, 2]
        x = x.clone()
        x[:, :1] = x[:, :1] * 2.0 - 1.0                # t: [0,1] -> [-1,1]
        x = x * self.input_gamma + self.input_beta
        out = self.net(x)                               # [B, 2]
        u_tpf = out[:, :1] + self.bias_u
        k     = self.softplus(out[:, 1:2])
        tpf = torch.sigmoid(u_tpf) * self.t_rng + self.t_min
        return tpf, k

# =========================
# Physics: d(1/v)/dt
# =========================
def d_inv_dt_fd(t_norm, v_raw, smooth_k=7, t_min=0.0, t_rng=1.0):
    t_minute = t_norm.squeeze(1) * t_rng + t_min
    invv = 1.0 / torch.clamp(v_raw, min=1e-6)
    y = invv.squeeze(1)
    y_s = moving_average(y, k=smooth_k)

    idx = torch.argsort(t_minute)
    ts  = t_minute[idx]
    ys  = y_s[idx]

    der = torch.zeros_like(ys)
    der[1:-1] = (ys[2:] - ys[:-2]) / torch.clamp(ts[2:] - ts[:-2], min=1e-6)
    der[0]    = (ys[1]  - ys[0])   / torch.clamp(ts[1]  - ts[0],   min=1e-6)
    der[-1]   = (ys[-1] - ys[-2])  / torch.clamp(ts[-1] - ts[-2],  min=1e-6)

    inv = torch.empty_like(idx); inv[idx] = torch.arange(len(idx), device=idx.device)
    return der[inv].unsqueeze(1)

# =========================
# Losses
# =========================
def compute_losses(model, t_norm, vhat_input, taf_min=None,
                   lam=0.10, smooth_k=9, alpha_var=0.00, beta_point=0.0,
                   v_raw_for_phys: torch.Tensor = None):
    tpf, k = model(t_norm, vhat_input)

    # 监督项：均值锚定 + 可选逐点
    if taf_min is not None:
        taf_t = torch.tensor([[taf_min]], dtype=torch.float32, device=tpf.device)
        L_D_mean = (tpf.mean() - taf_t.squeeze(0)).pow(2)
        if beta_point > 0:
            L_D_point = torch.mean((tpf - taf_t.expand_as(tpf))**2)
        else:
            L_D_point = torch.tensor(0.0, device=tpf.device)
        L_D = L_D_mean + beta_point * L_D_point
    else:
        L_D = torch.tensor(0.0, device=tpf.device)

    # 物理残差
    with torch.no_grad():
        if v_raw_for_phys is None:
            raise ValueError("v_raw_for_phys 不能为空")
        d_inv = d_inv_dt_fd(t_norm, v_raw_for_phys, smooth_k=smooth_k,
                            t_min=model.raw_t_min, t_rng=model.raw_t_rng)

    resid = d_inv + k
    med = torch.median(resid)
    mad = torch.median(torch.abs(resid - med)) + 1e-6
    resid_n = resid / mad
    L_P = torch.mean(resid_n**2)

    L_var = torch.var(tpf, unbiased=False) if alpha_var > 0 else torch.tensor(0.0, device=tpf.device)

    total = L_D + lam * L_P + alpha_var * L_var
    return total, L_D, L_P, L_var, tpf, k, d_inv

# =========================
# Train
# =========================
def train(model, t_norm, vhat, taf_min, lam=0.10, lr=2e-3, epochs=800,
          grad_thresh=1e-3, print_every=100, smooth_k=9, alpha_var=0.00, beta_point=0.0,
          v_raw_for_phys: torch.Tensor = None, clip_norm: float = 1.0):
    opt_adam  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    opt_lbfgs = optim.LBFGS(model.parameters(), max_iter=15, line_search_fn='strong_wolfe')

    warmup = min(200, epochs // 2)
    last_log = {}
    for ep in range(epochs):
        lam_ep = lam * (ep / warmup) if ep < warmup else lam

        opt_adam.zero_grad()
        tot, Ld, Lp, Lvar, _, _, _ = compute_losses(
            model, t_norm, vhat, taf_min,
            lam=lam_ep, smooth_k=smooth_k, alpha_var=alpha_var, beta_point=beta_point,
            v_raw_for_phys=v_raw_for_phys
        )
        tot.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        # grad norm
        g2 = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g2 += float(p.grad.data.norm(2)**2)
        gnorm = math.sqrt(g2)

        if gnorm > grad_thresh:
            opt_adam.step()
        else:
            def closure():
                opt_lbfgs.zero_grad()
                tot2, _, _, _, _, _, _ = compute_losses(
                    model, t_norm, vhat, taf_min,
                    lam=lam_ep, smooth_k=smooth_k, alpha_var=alpha_var, beta_point=beta_point,
                    v_raw_for_phys=v_raw_for_phys
                )
                tot2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                return tot2
            opt_lbfgs.step(closure)

        if ep % print_every == 0 or ep == epochs - 1:
            with torch.no_grad():
                _, Ld_, Lp_, Lvar_, tpf_, k_, d_inv_ = compute_losses(
                    model, t_norm, vhat, taf_min,
                    lam=lam_ep, smooth_k=smooth_k, alpha_var=alpha_var, beta_point=beta_point,
                    v_raw_for_phys=v_raw_for_phys
                )
                last_log = dict(
                    ep=ep,
                    total=float((Ld_ + lam_ep*Lp_ + alpha_var*Lvar_).item()),
                    L_D=float(Ld_.item()),
                    L_P=float(Lp_.item()),
                    L_var=float(Lvar_.item()),
                    gnorm=gnorm,
                    k_mean=float(k_.mean().item()),
                    d1v_mean=float(d_inv_.mean().item()),
                    d1v_std=float(d_inv_.std(unbiased=False).item()),
                    tpf_mean=float(tpf_.mean().item()),
                    lam_ep=float(lam_ep),
                )
                print(f"[{ep:4d}] Total={last_log['total']:.4e}  "
                      f"L_D={last_log['L_D']:.4e}  L_P={last_log['L_P']:.4e}  "
                      f"L_var={last_log['L_var']:.4e}  ||g||={gnorm:.3e}  "
                      f"tpf_mean={last_log['tpf_mean']:.2f}  "
                      f"k_mean={last_log['k_mean']:.3e}  "
                      f"d(1/v)/dt mean±std={last_log['d1v_mean']:.3e}±{last_log['d1v_std']:.3e}  "
                      f"lam={lam_ep:.3f}")
    return last_log

# =========================
# Main
# =========================
def main(xlsx, dt=7.0, lam=0.10, lr=2e-3, epochs=800, taf_minutes=None,
         seed=2025, use_cuda=False, smooth_k=9, alpha_var=0.00, beta_point=0.0,
         arch: str = "kan", kan_basis: int = 16, hidden: int = 64,
         margin_low_ratio: float = 0.10, margin_high_ratio: float = 0.10):
    set_seed(seed)


    t_np, v_np, t_last = load_series(xlsx, dt_min=dt)
    if len(t_np) < 3:
        raise ValueError(f"有效样本点太少（{len(t_np)}），物理差分至少需要 >=3 点；请检查 v>0 的过滤后数据。")

    taf = float(taf_minutes) if taf_minutes is not None else float(t_last)

    # 归一化输入
    t_min, t_max = float(t_np.min()), float(t_np.max())
    t_rng = max(t_max - t_min, 1e-6)
    t_norm_np = (t_np - t_min) / t_rng

    v_mean, v_std = float(v_np.mean()), float(v_np.std() or 1.0)
    vhat_np = (v_np - v_mean) / (v_std if v_std != 0 else 1.0)

    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    t_norm = torch.tensor(t_norm_np, dtype=torch.float32, device=device).unsqueeze(1)
    vhat   = torch.tensor(vhat_np,    dtype=torch.float32, device=device).unsqueeze(1)
    v_raw  = torch.tensor(v_np,       dtype=torch.float32, device=device).unsqueeze(1)


    model = PIKANModel(
        hidden=hidden, t_min=t_min, t_rng=t_rng, arch=arch, kan_basis=kan_basis,
        margin_low_ratio=margin_low_ratio, margin_high_ratio=margin_high_ratio
    ).to(device)


    log = train(model, t_norm, vhat, taf_min=taf, lam=lam, lr=lr, epochs=epochs,
                grad_thresh=1e-3, print_every=100, smooth_k=smooth_k,
                alpha_var=alpha_var, beta_point=beta_point,
                v_raw_for_phys=v_raw, clip_norm=1.0)


    with torch.no_grad():
        tpf_all, k_all = model(t_norm, vhat)
        tpf_avg = float(tpf_all.mean().item())
        k_avg   = float(k_all.mean().item())

    t_total = float(t_max - t_min)
    RE = abs(tpf_avg - taf) / (t_total if t_total > 0 else 1.0) * 100.0

    df_out = pd.DataFrame([{
        "N_points": len(t_np),
        "dt_min": dt,
        "taf_min": taf,
        "t_window_min": t_total,
        "tpf_avg_min": tpf_avg,
        "k_avg": k_avg,
        "RE_percent": RE,
        "k_mean_log": log.get("k_mean", np.nan),
        "d1v_mean": log.get("d1v_mean", np.nan),
        "d1v_std": log.get("d1v_std", np.nan),
        "arch": arch,
        "kan_basis": kan_basis,
        "hidden": hidden,
        "margin_low_ratio": margin_low_ratio,
        "margin_high_ratio": margin_high_ratio,
        "beta_point": beta_point,
    }])

    title = f"PIKAN({arch.upper()}) 训练结果 (taf={int(round(taf))} min)"
    display_dataframe_to_user(title, df_out)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = f"pikans_{arch}_taf{int(round(taf))}_{stamp}.csv"
    df_out.to_csv(out_csv, index=False)

    print("\n=== Results ===")
    print(df_out.to_string(index=False))
    print(f"\nSaved results to: {out_csv}")


main("test.xlsx",
     dt=7.0,
     lam=0.10,
     lr=2e-3,
     epochs=800,
     taf_minutes=1470.0,
     seed=2025,
     use_cuda=False,
     smooth_k=9,
     alpha_var=0.00,
     beta_point=0.0,
     arch="kan",
     kan_basis=16,
     hidden=64,
     margin_low_ratio=0.10,
     margin_high_ratio=0.10)
