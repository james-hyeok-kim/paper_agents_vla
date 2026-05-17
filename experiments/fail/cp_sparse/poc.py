"""
CP-Sparse PoC: chunk-position cross-attention entropy in a tiny ACT-like model.

Critical question: Does AutoHorizon's "intra-chunk attention invariance" generalize
to ACT-style models? Specifically, do later chunk positions have:
  A. Lower entropy than earlier? (slope > 0.3 nats/position?)
  B. Higher top-k mass? (k=8 mass > 0.8 at late positions?)
  C. Different argmax sets from position 0? (overlap < 0.7 means GO, > 0.95 means NO-GO)

Approach: train tiny ACT-like model on synthetic chunk-prediction task for ~3 min,
then measure cross-attention statistics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


# ---------- Tiny ACT-like model ----------

class TinyACT(nn.Module):
    """
    Encoder: takes a context sequence (representing visual+state features)
    Decoder: chunk_len learnable positional queries cross-attending to encoder memory.
    Outputs an action chunk of length chunk_len.
    """
    def __init__(self, d_model=128, n_heads=4, n_enc=2, n_dec=2, ctx_len=64, chunk_len=50, action_dim=7):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ctx_len = ctx_len
        self.chunk_len = chunk_len

        # Context (memory) projection — learnable from raw input
        self.ctx_in = nn.Linear(d_model, d_model)
        self.ctx_pos = nn.Parameter(torch.randn(ctx_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)

        # Decoder: learnable position queries
        self.pos_queries = nn.Parameter(torch.randn(chunk_len, d_model) * 0.02)
        # Use our own decoder that exposes cross-attention weights
        self.dec_layers = nn.ModuleList([
            CrossAttnDecoderLayer(d_model, n_heads) for _ in range(n_dec)
        ])
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, ctx, return_attn=False):
        # ctx: (B, ctx_len, d_model) raw context
        B = ctx.size(0)
        mem = self.encoder(self.ctx_in(ctx) + self.ctx_pos.unsqueeze(0))  # (B, ctx_len, d)
        # Decoder queries
        q = self.pos_queries.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, chunk_len, d)
        all_attn = []
        for layer in self.dec_layers:
            q, attn = layer(q, mem, return_attn=True)
            all_attn.append(attn)  # (B, n_heads, chunk_len, ctx_len)
        out = self.action_head(q)  # (B, chunk_len, action_dim)
        if return_attn:
            return out, all_attn
        return out


class CrossAttnDecoderLayer(nn.Module):
    """Cross-attention only decoder layer that exposes attention weights."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Self-attn on queries
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Cross-attn
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_k = nn.Linear(d_model, d_model)
        self.cross_v = nn.Linear(d_model, d_model)
        self.cross_out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, q, mem, return_attn=False):
        # Self-attn
        sa, _ = self.self_attn(q, q, q)
        q = self.norm1(q + sa)
        # Cross-attn (manual to expose weights)
        B, T, D = q.shape
        S = mem.size(1)
        H = self.n_heads
        Hd = self.d_head
        Q = self.cross_q(q).reshape(B, T, H, Hd).transpose(1, 2)  # (B, H, T, Hd)
        K = self.cross_k(mem).reshape(B, S, H, Hd).transpose(1, 2)  # (B, H, S, Hd)
        V = self.cross_v(mem).reshape(B, S, H, Hd).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Hd)  # (B, H, T, S)
        attn = F.softmax(scores, dim=-1)  # (B, H, T, S)
        ctx = torch.matmul(attn, V)  # (B, H, T, Hd)
        ctx = ctx.transpose(1, 2).reshape(B, T, D)
        ca = self.cross_out(ctx)
        q = self.norm2(q + ca)
        # FF
        q = self.norm3(q + self.ff(q))
        if return_attn:
            return q, attn
        return q


# ---------- Synthetic data ----------

def make_synthetic_data(n_episodes=400, chunk_len=50, ctx_len=64, d=128, action_dim=7):
    """
    Each "scene" has a context vector that determines target actions.
    To make late-position prediction harder than just copying early,
    we use a function where:
       target_action[t] = f(ctx[:context_t_idx]) where context_t_idx grows with t
    so late actions depend on a wider window of context.

    Then we test: does the trained model concentrate attention or spread it?
    """
    rng = np.random.RandomState(0)
    # Context: (n_episodes, ctx_len, d) random vectors
    ctx = torch.randn(n_episodes, ctx_len, d, device=device)
    # Actions: target_action[t] = ctx[map(t)] projected to action_dim + smooth
    # We use a fixed projection
    proj = torch.randn(d, action_dim, device=device) * 0.5
    actions = []
    for t in range(chunk_len):
        # Late positions depend on later context tokens
        # Position 0 depends on ctx[0]; position chunk_len-1 depends on ctx[ctx_len-1]
        idx = int((t / max(1, chunk_len-1)) * (ctx_len - 1))
        a_t = ctx[:, idx, :] @ proj  # (n_episodes, action_dim)
        # Add small noise so model has to actually learn the mapping
        a_t = a_t + 0.01 * torch.randn_like(a_t)
        actions.append(a_t)
    actions = torch.stack(actions, dim=1)  # (n_episodes, chunk_len, action_dim)
    return ctx, actions


def train_tiny_act(d=128, chunk_len=50, ctx_len=64, n_iters=500, batch_size=64):
    model = TinyACT(d_model=d, chunk_len=chunk_len, ctx_len=ctx_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    ctx, target_actions = make_synthetic_data(n_episodes=2000, chunk_len=chunk_len,
                                              ctx_len=ctx_len, d=d)
    N = ctx.size(0)

    print(f"Training tiny ACT: chunk_len={chunk_len}, ctx_len={ctx_len}, d_model={d}")
    losses = []
    for it in range(n_iters):
        idx = torch.randint(0, N, (batch_size,), device=device)
        b_ctx = ctx[idx]
        b_tgt = target_actions[idx]
        pred = model(b_ctx)
        loss = ((pred - b_tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (it + 1) % 100 == 0:
            recent = np.mean(losses[-50:])
            print(f"  iter {it+1}: loss={recent:.5f}")
    return model, ctx, target_actions


def analyze_attention(model, ctx, target_actions, chunk_len, ctx_len, n_samples=256):
    """Return per-position entropy, top-k mass, argmax overlap stats."""
    model.eval()
    with torch.no_grad():
        b_ctx = ctx[:n_samples]
        b_tgt = target_actions[:n_samples]
        pred, all_attn = model(b_ctx, return_attn=True)
        # final-loss check
        final_loss = ((pred - b_tgt) ** 2).mean().item()
        # Average attention over the last decoder layer (most relevant), and over heads
        # all_attn[-1]: (B, H, chunk_len, ctx_len)
        attn = all_attn[-1].mean(dim=1)  # (B, chunk_len, ctx_len)
        # Also try first decoder layer
        attn_first = all_attn[0].mean(dim=1)

    # ===== Gate A: per-position entropy =====
    eps = 1e-12
    entropy = -(attn * (attn + eps).log()).sum(-1)  # (B, chunk_len) in nats
    entropy_mean = entropy.mean(0).cpu().numpy()  # (chunk_len,)
    # Slope: linear regression entropy vs position
    pos = np.arange(chunk_len)
    slope = float(np.polyfit(pos, entropy_mean, 1)[0])
    print(f"  Final training loss: {final_loss:.5f}")
    print(f"  Entropy at position 0: {entropy_mean[0]:.3f} nats")
    print(f"  Entropy at position {chunk_len//2}: {entropy_mean[chunk_len//2]:.3f} nats")
    print(f"  Entropy at position {chunk_len-1}: {entropy_mean[-1]:.3f} nats")
    print(f"  Slope (linear fit): {slope:.4f} nats/position")
    # Max-min entropy
    print(f"  Entropy range: [{entropy_mean.min():.3f}, {entropy_mean.max():.3f}]")

    # ===== Gate B: top-k mass =====
    K_VALUES = [4, 8, 16, 32]
    topk_mass = {}
    for k in K_VALUES:
        kk = min(k, ctx_len)
        topk_v, _ = attn.topk(kk, dim=-1)  # (B, chunk_len, k)
        mass = topk_v.sum(-1).mean(0).cpu().numpy()  # (chunk_len,)
        topk_mass[k] = {"by_pos": mass.tolist(),
                        "pos0": float(mass[0]),
                        "pos_mid": float(mass[chunk_len//2]),
                        "pos_last": float(mass[-1]),
                        "max_in_late_half": float(mass[chunk_len//2:].max()),
                        "min_in_late_half": float(mass[chunk_len//2:].min())}
        print(f"  Top-{k} mass:  pos0={mass[0]:.3f}, mid={mass[chunk_len//2]:.3f}, last={mass[-1]:.3f}")

    # ===== Gate C: argmax overlap =====
    # Per sample, get top-k indices at position 0 and at position chunk_len-1, measure Jaccard
    JACCARD_KS = [4, 8, 16]
    jaccard = {}
    for jk in JACCARD_KS:
        _, top_idx_0 = attn[:, 0, :].topk(min(jk, ctx_len), dim=-1)  # (B, jk)
        _, top_idx_last = attn[:, -1, :].topk(min(jk, ctx_len), dim=-1)
        jac_per_sample = []
        for b in range(top_idx_0.size(0)):
            s0 = set(top_idx_0[b].tolist())
            sl = set(top_idx_last[b].tolist())
            j = len(s0 & sl) / len(s0 | sl)
            jac_per_sample.append(j)
        jac_mean = float(np.mean(jac_per_sample))
        jaccard[jk] = jac_mean
        print(f"  Argmax overlap (top-{jk}) pos0 vs pos{chunk_len-1}: Jaccard={jac_mean:.3f}")

    # Also for position 0 vs mid
    _, top_idx_0 = attn[:, 0, :].topk(8, dim=-1)
    _, top_idx_mid = attn[:, chunk_len//2, :].topk(8, dim=-1)
    jac_mid = float(np.mean([len(set(top_idx_0[b].tolist()) & set(top_idx_mid[b].tolist())) /
                             len(set(top_idx_0[b].tolist()) | set(top_idx_mid[b].tolist()))
                             for b in range(top_idx_0.size(0))]))
    print(f"  Argmax overlap (top-8) pos0 vs pos{chunk_len//2}: Jaccard={jac_mid:.3f}")

    return {
        "final_loss": final_loss,
        "entropy_by_pos": entropy_mean.tolist(),
        "entropy_slope_nats_per_pos": slope,
        "entropy_pos0": float(entropy_mean[0]),
        "entropy_pos_mid": float(entropy_mean[chunk_len//2]),
        "entropy_pos_last": float(entropy_mean[-1]),
        "topk_mass": topk_mass,
        "argmax_jaccard_0_vs_last": jaccard,
        "argmax_jaccard_0_vs_mid_top8": jac_mid,
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("CP-Sparse PoC: tiny ACT-like model cross-attention analysis")
    print(f"Device: {device}")
    print("=" * 60)

    chunk_len = 50
    ctx_len = 64
    d_model = 128

    print("\n[1/2] Training tiny ACT...")
    model, ctx, tgt = train_tiny_act(d=d_model, chunk_len=chunk_len, ctx_len=ctx_len,
                                     n_iters=500, batch_size=64)

    print("\n[2/2] Analyzing cross-attention...")
    stats = analyze_attention(model, ctx, tgt, chunk_len=chunk_len, ctx_len=ctx_len)

    elapsed = time.time() - t0
    print(f"\n=== Elapsed: {elapsed:.1f}s ===")

    # ===== Gate evaluation =====
    print("\n" + "=" * 60)
    print("GATE EVALUATION")
    print("=" * 60)
    # Gate A: |slope| > 0.3 nats/position (positive or negative)
    slope = stats["entropy_slope_nats_per_pos"]
    gate_a = abs(slope) > 0.3 / 50  # 0.3 nats over 50 positions = 0.006 nats/pos
    # Note: "slope > 0.3 nats/position" was a typo in the spec; clearer: total entropy change > 0.3 nats end-to-end
    total_change = stats["entropy_pos_last"] - stats["entropy_pos0"]
    gate_a_alt = abs(total_change) > 0.3

    # Gate B: top-8 mass > 0.8 at late positions
    late_max = stats["topk_mass"][8]["max_in_late_half"]
    late_min = stats["topk_mass"][8]["min_in_late_half"]
    gate_b = late_max > 0.8

    # Gate C: argmax overlap (top-8) pos0 vs pos_last
    jac = stats["argmax_jaccard_0_vs_last"][8]
    if jac > 0.95:
        gate_c_class = "NO-GO (AutoHorizon invariance holds — reframe to anchor-centered uniform)"
    elif jac > 0.7:
        gate_c_class = "CONDITIONAL (union-of-top-k needed)"
    else:
        gate_c_class = "GO (first-position argmax reusable)"

    print(f"\nGate A (entropy slope): |slope|={abs(slope):.5f} nats/pos, "
          f"end-to-end Δ entropy = {total_change:+.3f} nats")
    print(f"   → {'PASS' if gate_a_alt else 'FAIL'} ({gate_a_alt})")
    print(f"\nGate B (top-8 mass at late half): max={late_max:.3f}, min={late_min:.3f}")
    print(f"   → {'PASS' if gate_b else 'FAIL'} (target: >0.8)")
    print(f"\nGate C (argmax top-8 overlap pos0 vs pos{chunk_len-1}): Jaccard={jac:.3f}")
    print(f"   → {gate_c_class}")

    # Overall
    overall = gate_a_alt and gate_b and jac < 0.95
    print(f"\nOverall: {'PASS' if overall else 'FAIL/CONDITIONAL'}")
    print(f"AutoHorizon invariance holds for this tiny ACT? {'YES' if jac > 0.95 else 'NO'}")

    out = {
        "elapsed_sec": elapsed,
        "device": str(device),
        "config": {"chunk_len": chunk_len, "ctx_len": ctx_len, "d_model": d_model},
        "stats": stats,
        "gates": {
            "A_entropy_slope_pass": bool(gate_a_alt),
            "A_total_entropy_change_nats": float(total_change),
            "A_slope_nats_per_pos": float(slope),
            "B_topk8_late_half_max_mass": float(late_max),
            "B_pass": bool(gate_b),
            "C_argmax_jaccard_top8_0_vs_last": float(jac),
            "C_classification": gate_c_class,
        },
        "verdict": ("CP-Sparse FEASIBILITY: PASS" if overall
                    else f"CP-Sparse FEASIBILITY: {gate_c_class}"),
    }
    with open("/home/jovyan/workspace/paper_agents_vla/experiments/cp_sparse/results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to results.json")


if __name__ == "__main__":
    main()
