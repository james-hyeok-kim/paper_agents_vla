"""
AMP-Distill PoC: SO(3) geodesic vs L2 quaternion loss.

Key question: Does SO(3) geodesic loss + contact-phase reweighting give different
gradient behavior than L2 on quaternions in contact-rich vs contact-poor settings?

Synthetic data only. Self-contained. Targets ~2 min.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ---------- SE(3) utilities (manual PyTorch impl, no roma) ----------

def quat_normalize(q):
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)

def quaternion_to_R(q):
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1),
    ], dim=-2)
    return R

def so3_geodesic(R1, R2):
    """Rotation geodesic in radians."""
    rel = torch.matmul(R1.transpose(-2, -1), R2)
    trace = torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos)

def quat_l2(q1, q2):
    """L2 quaternion loss with double-cover fix."""
    # Pick closer hemisphere
    sign = torch.sign((q1 * q2).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q2_aligned = q2 * sign
    return ((q1 - q2_aligned) ** 2).sum(-1)

def quat_l2_naive(q1, q2):
    """Naive L2 quaternion loss (no double-cover fix) — what ActDistill uses."""
    return ((q1 - q2) ** 2).sum(-1)

# ---------- Teacher trajectories ----------

def make_random_quat(n):
    q = torch.randn(n, 4, device=device)
    return quat_normalize(q)

def axis_angle_to_quat(axis, angle):
    """axis: (..., 3), angle: (...,) -> (..., 4) wxyz"""
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    half = angle / 2
    sh = torch.sin(half)
    return torch.stack([torch.cos(half), axis[..., 0]*sh, axis[..., 1]*sh, axis[..., 2]*sh], -1)

def generate_teacher_trajectory(T=50, contact_rich=False):
    """
    contact_rich=False: translation dominant, small rotation (reach-and-move)
    contact_rich=True:  rotation dominant in last 40% (peg-in-hole, screwing)
    """
    # Translation: smooth
    trans = torch.cumsum(torch.randn(T, 3, device=device) * 0.02, dim=0)
    # Rotation
    quats = [make_random_quat(1).squeeze(0)]
    if not contact_rich:
        # Small per-step rotation
        for t in range(1, T):
            axis = torch.randn(3, device=device)
            angle = torch.tensor(0.02, device=device)
            dq = axis_angle_to_quat(axis, angle)
            q_new = quat_normalize(quats[-1] + dq * 0.05)  # blend
            quats.append(q_new)
    else:
        # Small rotation first half, large rotation second half
        for t in range(1, T):
            axis = torch.randn(3, device=device)
            angle = torch.tensor(0.02 if t < T // 2 else 0.25, device=device)
            dq = axis_angle_to_quat(axis, angle)
            # Compose: q_new = dq * q_old
            q_new = quat_mul(dq, quats[-1])
            quats.append(q_new)
    quat = torch.stack(quats, dim=0)
    gripper = (torch.rand(T, device=device) > 0.5).float()
    contact_mask = torch.zeros(T, device=device)
    if contact_rich:
        contact_mask[T // 2:] = 1.0
    return trans, quat, gripper, contact_mask

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

# ---------- Student model (tiny MLP, trained to match teacher) ----------

class StudentMLP(nn.Module):
    """Tiny student that predicts (trans, quat, gripper) from a small input."""
    def __init__(self, T=50):
        super().__init__()
        self.T = T
        # Input is teacher's noisy observation; predict full trajectory.
        self.net = nn.Sequential(
            nn.Linear(T * 4, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.trans_head = nn.Linear(256, T * 3)
        self.quat_head = nn.Linear(256, T * 4)
        self.grip_head = nn.Linear(256, T)

    def forward(self, obs):
        # obs: (B, T*4) noisy teacher action summary
        h = self.net(obs)
        T = self.T
        trans = self.trans_head(h).view(-1, T, 3)
        quat = quat_normalize(self.quat_head(h).view(-1, T, 4))
        grip = self.grip_head(h).view(-1, T)
        return trans, quat, grip

# ---------- Training procedures ----------

def train_student(loss_name, contact_rich, n_episodes=200, n_iters=300, T=50, contact_reweight=False):
    """
    loss_name: 'l2_naive' (ActDistill-style) or 'l2_fixed' or 'so3_geodesic'
    Returns final translation MSE, rotation geodesic, gripper acc, separately per phase.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    # Pre-generate dataset
    trajs = [generate_teacher_trajectory(T, contact_rich=contact_rich) for _ in range(n_episodes)]
    obs_list = []
    for trans, quat, grip, _ in trajs:
        # Observation = teacher actions + noise (simulates "what student must regress")
        obs = torch.cat([
            trans.flatten(),
            quat.flatten(),
            grip.flatten().unsqueeze(-1).expand(-1, 1).flatten()
        ]).unsqueeze(0)
        # Crop to T*4
        obs = obs[:, :T*4] + 0.1 * torch.randn(1, T*4, device=device)
        obs_list.append(obs)
    obs = torch.cat(obs_list, dim=0)  # (N, T*4)
    teacher_trans = torch.stack([t[0] for t in trajs], dim=0)
    teacher_quat = torch.stack([t[1] for t in trajs], dim=0)
    teacher_grip = torch.stack([t[2] for t in trajs], dim=0)
    contact_mask = torch.stack([t[3] for t in trajs], dim=0)  # (N, T)

    model = StudentMLP(T=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for it in range(n_iters):
        opt.zero_grad()
        pred_trans, pred_quat, pred_grip = model(obs)
        # Translation loss: L2
        L_trans = ((pred_trans - teacher_trans) ** 2).sum(-1)  # (N, T)
        # Rotation loss
        if loss_name == 'l2_naive':
            L_rot = quat_l2_naive(pred_quat, teacher_quat)  # (N, T)
        elif loss_name == 'l2_fixed':
            L_rot = quat_l2(pred_quat, teacher_quat)
        elif loss_name == 'so3_geodesic':
            Rs = quaternion_to_R(pred_quat)  # (N, T, 3, 3)
            Rt = quaternion_to_R(teacher_quat)
            L_rot = so3_geodesic(Rs, Rt) ** 2
        else:
            raise ValueError(loss_name)
        L_grip = F.binary_cross_entropy_with_logits(pred_grip, teacher_grip, reduction='none')

        # Contact reweight (only for proposed method)
        if contact_reweight:
            # Upweight rotation loss in contact phase by 3x
            w = 1.0 + 2.0 * contact_mask  # (N, T)
            L_rot = L_rot * w
            L_trans = L_trans * w

        loss = L_trans.mean() + L_rot.mean() + 0.1 * L_grip.mean()
        loss.backward()
        opt.step()

    # Evaluate per-phase
    with torch.no_grad():
        pred_trans, pred_quat, pred_grip = model(obs)
        # Rotation geodesic error (in radians)
        Rs = quaternion_to_R(pred_quat)
        Rt = quaternion_to_R(teacher_quat)
        rot_geodesic = so3_geodesic(Rs, Rt)  # (N, T) in radians
        trans_mse = ((pred_trans - teacher_trans) ** 2).sum(-1).mean(0)  # (T,)
        grip_acc = ((pred_grip > 0).float() == teacher_grip).float().mean(0)  # (T,)

        if contact_rich:
            mask_c = contact_mask[0] > 0.5
            rot_contact = rot_geodesic[:, mask_c].mean().item()
            rot_free = rot_geodesic[:, ~mask_c].mean().item()
            trans_contact = trans_mse[mask_c].mean().item()
            trans_free = trans_mse[~mask_c].mean().item()
        else:
            rot_contact = float('nan')
            rot_free = rot_geodesic.mean().item()
            trans_contact = float('nan')
            trans_free = trans_mse.mean().item()
        rot_overall = rot_geodesic.mean().item()
        trans_overall = trans_mse.mean().item()
        grip_overall = grip_acc.mean().item()

    return {
        "rot_overall_rad": rot_overall,
        "rot_contact_rad": rot_contact,
        "rot_free_rad": rot_free,
        "trans_overall_mse": trans_overall,
        "trans_contact_mse": trans_contact,
        "trans_free_mse": trans_free,
        "grip_acc": grip_overall,
    }


def main():
    results = {}
    t0 = time.time()

    print("=" * 60)
    print("AMP-Distill PoC: synthetic SE(3) distillation")
    print(f"Device: {device}, T=50, n_episodes=200, n_iters=300")
    print("=" * 60)

    # Setting 1: Contact-rich trajectory (peg-in-hole style)
    print("\n[Setting 1] Contact-rich trajectory")
    for loss_name in ['l2_naive', 'l2_fixed', 'so3_geodesic']:
        for cr in [False, True]:
            tag = f"contact_rich__{loss_name}__reweight_{cr}"
            r = train_student(loss_name, contact_rich=True, contact_reweight=cr)
            results[tag] = r
            print(f"  [{tag}]")
            print(f"    rot_overall={np.degrees(r['rot_overall_rad']):.2f}deg, "
                  f"rot_contact={np.degrees(r['rot_contact_rad']):.2f}deg, "
                  f"rot_free={np.degrees(r['rot_free_rad']):.2f}deg, "
                  f"trans={r['trans_overall_mse']:.4f}, grip={r['grip_acc']:.3f}")

    # Setting 2: Contact-poor trajectory (translation-dominant)
    print("\n[Setting 2] Contact-poor trajectory")
    for loss_name in ['l2_naive', 'l2_fixed', 'so3_geodesic']:
        for cr in [False, True]:
            tag = f"contact_poor__{loss_name}__reweight_{cr}"
            r = train_student(loss_name, contact_rich=False, contact_reweight=cr)
            results[tag] = r
            print(f"  [{tag}]")
            print(f"    rot_overall={np.degrees(r['rot_overall_rad']):.2f}deg, "
                  f"trans={r['trans_overall_mse']:.4f}, grip={r['grip_acc']:.3f}")

    elapsed = time.time() - t0
    print(f"\n=== Elapsed: {elapsed:.1f}s ===")

    # ===== Decision logic =====
    print("\n" + "=" * 60)
    print("GATE EVALUATION")
    print("=" * 60)

    # Compare SO(3) geodesic vs L2 naive on contact-rich trajectory
    r_so3 = results["contact_rich__so3_geodesic__reweight_True"]
    r_l2 = results["contact_rich__l2_naive__reweight_False"]
    r_so3_no_rw = results["contact_rich__so3_geodesic__reweight_False"]
    r_l2_fixed = results["contact_rich__l2_fixed__reweight_False"]

    # In contact phase (where rotation matters)
    rot_so3_contact_deg = np.degrees(r_so3['rot_contact_rad'])
    rot_l2_contact_deg = np.degrees(r_l2['rot_contact_rad'])
    delta_contact_deg = rot_l2_contact_deg - rot_so3_contact_deg

    rot_so3_free_deg = np.degrees(r_so3['rot_free_rad'])
    rot_l2_free_deg = np.degrees(r_l2['rot_free_rad'])
    delta_free_deg = rot_l2_free_deg - rot_so3_free_deg

    # Contact-poor
    rot_so3_poor_deg = np.degrees(results["contact_poor__so3_geodesic__reweight_True"]['rot_overall_rad'])
    rot_l2_poor_deg = np.degrees(results["contact_poor__l2_naive__reweight_False"]['rot_overall_rad'])
    delta_poor_deg = rot_l2_poor_deg - rot_so3_poor_deg

    print(f"\nRotation error (deg) in CONTACT phase of contact-rich trajectory:")
    print(f"  L2-naive            : {rot_l2_contact_deg:.3f}")
    print(f"  SO(3)+reweight      : {rot_so3_contact_deg:.3f}")
    print(f"  Improvement (L2 - SO3)         = {delta_contact_deg:.3f} deg")

    print(f"\nRotation error (deg) in FREE phase of contact-rich trajectory:")
    print(f"  L2-naive            : {rot_l2_free_deg:.3f}")
    print(f"  SO(3)+reweight      : {rot_so3_free_deg:.3f}")
    print(f"  Improvement                      = {delta_free_deg:.3f} deg")

    print(f"\nRotation error (deg) in CONTACT-POOR trajectory (whole):")
    print(f"  L2-naive            : {rot_l2_poor_deg:.3f}")
    print(f"  SO(3)+reweight      : {rot_so3_poor_deg:.3f}")
    print(f"  Improvement                      = {delta_poor_deg:.3f} deg")

    # Gate: SO(3) should give larger improvement in contact-rich than contact-poor
    print("\n--- GATE ---")
    print(f"  Contact-rich gain : {delta_contact_deg:.3f} deg")
    print(f"  Contact-poor gain : {delta_poor_deg:.3f} deg")
    print(f"  Specificity ratio : {delta_contact_deg / (delta_poor_deg + 1e-6):.2f}x")

    # Also check L2 double-cover problem
    print(f"\n--- Double-cover check ---")
    print(f"  L2 naive  vs L2 fixed (contact phase): "
          f"{rot_l2_contact_deg:.3f} vs {np.degrees(r_l2_fixed['rot_contact_rad']):.3f} deg")

    pass_gate_a = delta_contact_deg > 0  # SO(3) helps in contact phase
    pass_gate_b = delta_contact_deg > delta_poor_deg  # Specificity
    if pass_gate_a and pass_gate_b:
        verdict = "FEASIBILITY: PASS — mechanism specificity confirmed in synthetic data"
    elif pass_gate_a:
        verdict = "FEASIBILITY: PARTIAL — SO(3) helps globally but not specific to contact phase"
    else:
        verdict = "FEASIBILITY: FAIL — SO(3) does not improve over L2 in synthetic data"
    print(f"\n>>> {verdict} <<<")

    # Save
    out = {
        "elapsed_sec": elapsed,
        "device": str(device),
        "results": {k: {k2: float(v2) if not (isinstance(v2,float) and np.isnan(v2)) else None
                        for k2, v2 in v.items()} for k, v in results.items()},
        "deltas": {
            "contact_rich_contact_phase_deg": float(delta_contact_deg),
            "contact_rich_free_phase_deg": float(delta_free_deg),
            "contact_poor_overall_deg": float(delta_poor_deg),
            "specificity_ratio": float(delta_contact_deg / (delta_poor_deg + 1e-6)),
        },
        "verdict": verdict,
        "pass_gate_a_so3_helps_contact": bool(pass_gate_a),
        "pass_gate_b_specificity": bool(pass_gate_b),
    }
    with open("/home/jovyan/workspace/paper_agents_vla/experiments/amp_distill/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to results.json")

if __name__ == "__main__":
    main()
