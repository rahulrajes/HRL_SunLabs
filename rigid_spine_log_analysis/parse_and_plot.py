"""
compare_rb_vs_squad.py
Parses Isaac Lab RL training logs for RB and SQUAD runs,
saves CSVs, and generates comparison plots for a research poster.

USAGE:
    1. Set the two log paths below.
    2. Run: python compare_rb_vs_squad.py
    3. Outputs land in  ./outputs/  (created automatically).
"""

import re
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ── CONFIG — set your log file paths here ─────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # folder this script lives in
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RB_LOG    = BASE_DIR / "/Users/rahulrajesh/Desktop/rigid_spine_log_analysis/output_RB_4_3.txt"     
SQUAD_LOG = BASE_DIR / "/Users/rahulrajesh/Desktop/rigid_spine_log_analysis/output_squad_updated_rewards.txt" 

# ── 1. Copy originals to outputs for safe-keeping (never touch originals) ─────
for src in [RB_LOG, SQUAD_LOG]:
    if src.exists():
        dst = OUTPUT_DIR / ("original_" + src.name)
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"Backed up {src.name} → outputs/original_{src.name}")

# ── 2. Regex patterns ─────────────────────────────────────────────────────────
ITER_PAT = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")

METRICS = {
    "mean_reward":           re.compile(r"Mean reward\s*:\s*([-\d.]+)"),
    "mean_episode_length":   re.compile(r"Mean episode length\s*:\s*([-\d.]+)"),
    "value_function_loss":   re.compile(r"Mean value_function loss\s*:\s*([-\d.]+)"),
    "surrogate_loss":        re.compile(r"Mean surrogate loss\s*:\s*([-\d.]+)"),
    "entropy_loss":          re.compile(r"Mean entropy loss\s*:\s*([-\d.]+)"),
    "action_noise_std":      re.compile(r"Mean action noise std\s*:\s*([-\d.]+)"),
    "total_timesteps":       re.compile(r"Total timesteps\s*:\s*([\d]+)"),
    "rew_air_time":          re.compile(r"Episode_Reward/air_time\s*:\s*([-\d.]+)"),
    "rew_base_angular_vel":  re.compile(r"Episode_Reward/base_angular_velocity\s*:\s*([-\d.]+)"),
    "rew_base_linear_vel":   re.compile(r"Episode_Reward/base_linear_velocity\s*:\s*([-\d.]+)"),
    "rew_foot_clearance":    re.compile(r"Episode_Reward/foot_clearance\s*:\s*([-\d.]+)"),
    "rew_gait":              re.compile(r"Episode_Reward/gait\s*:\s*([-\d.]+)"),
    "rew_action_smoothness": re.compile(r"Episode_Reward/action_smoothness\s*:\s*([-\d.]+)"),
    "rew_air_time_variance": re.compile(r"Episode_Reward/air_time_variance\s*:\s*([-\d.]+)"),
    "rew_base_motion":       re.compile(r"Episode_Reward/base_motion\s*:\s*([-\d.]+)"),
    "rew_base_orientation":  re.compile(r"Episode_Reward/base_orientation\s*:\s*([-\d.]+)"),
    "rew_foot_slip":         re.compile(r"Episode_Reward/foot_slip\s*:\s*([-\d.]+)"),
    "rew_joint_acc":         re.compile(r"Episode_Reward/joint_acc\s*:\s*([-\d.]+)"),
    "rew_joint_pos":         re.compile(r"Episode_Reward/joint_pos\s*:\s*([-\d.]+)"),
    "rew_joint_torques":     re.compile(r"Episode_Reward/joint_torques\s*:\s*([-\d.]+)"),
    "rew_joint_vel":         re.compile(r"Episode_Reward/joint_vel\s*:\s*([-\d.]+)"),
    "terrain_level":         re.compile(r"Curriculum/terrain_levels\s*:\s*([-\d.]+)"),
    "error_vel_xy":          re.compile(r"Metrics/base_velocity/error_vel_xy\s*:\s*([-\d.]+)"),
    "error_vel_yaw":         re.compile(r"Metrics/base_velocity/error_vel_yaw\s*:\s*([-\d.]+)"),
    "term_timeout":          re.compile(r"Episode_Termination/time_out\s*:\s*([-\d.]+)"),
    "term_body_contact":     re.compile(r"Episode_Termination/body_contact\s*:\s*([-\d.]+)"),
    "term_out_of_bounds":    re.compile(r"Episode_Termination/terrain_out_of_bounds\s*:\s*([-\d.]+)"),
}

# ── 3. Parser ─────────────────────────────────────────────────────────────────
def parse_log(path: Path) -> pd.DataFrame:
    text    = path.read_bytes().decode("utf-16")
    lines   = text.splitlines()
    records = []
    current = {}
    for line in lines:
        m = ITER_PAT.search(line)
        if m:
            if current:
                records.append(current)
            current = {"iteration": int(m.group(1))}
            continue
        if current:
            for key, pat in METRICS.items():
                mm = pat.search(line)
                if mm and key not in current:
                    current[key] = float(mm.group(1))
    if current:
        records.append(current)
    df = pd.DataFrame(records)
    print(f"  {path.name}: {len(df)} iterations parsed")
    return df

print("Parsing logs...")
rb    = parse_log(RB_LOG)
squad = parse_log(SQUAD_LOG)

# Save CSVs (outputs folder only — originals untouched)
rb.to_csv(OUTPUT_DIR    / "data_RB.csv",    index=False)
squad.to_csv(OUTPUT_DIR / "data_SQUAD.csv", index=False)
print("CSVs saved to outputs/")

# ── 4. Smoothing helper ───────────────────────────────────────────────────────
def smooth(series, w=30):
    return series.rolling(window=w, min_periods=1, center=True).mean()

# ── 5. Plotting ───────────────────────────────────────────────────────────────
RB_COLOR    = "#E05C5C"   # red-ish
SQUAD_COLOR = "#4A90D9"   # blue-ish
ALPHA_RAW   = 0.18
LW          = 2.0

def dual(ax, col, label_rb="RB", label_sq="SQUAD", ylabel=None, title=None, smooth_w=30):
    """Plot one metric for both runs on the same axes."""
    for df, color, label in [(rb, RB_COLOR, label_rb), (squad, SQUAD_COLOR, label_sq)]:
        if col not in df.columns:
            continue
        itr = df["iteration"]
        s   = smooth(df[col], smooth_w)
        ax.plot(itr, df[col], color=color, alpha=ALPHA_RAW, linewidth=0.5)
        ax.plot(itr, s,       color=color, linewidth=LW, label=label)
    ax.set_title(title or col, fontsize=10, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

# ── Figure 1: Poster headline plots (2×2) ────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("RB vs SQUAD — Learning & Locomotion Comparison", fontsize=15, fontweight="bold")
fig1.subplots_adjust(hspace=0.4, wspace=0.3)

dual(axes[0,0], "mean_reward",       "RB", "SQUAD", "Reward",     "Mean Reward per Iteration")
dual(axes[0,1], "terrain_level",     "RB", "SQUAD", "Avg Level",  "Terrain Curriculum Level")
dual(axes[1,0], "rew_base_linear_vel","RB","SQUAD", "Reward",     "Linear Velocity Reward (Speed)")
dual(axes[1,1], "error_vel_xy",      "RB", "SQUAD", "Error",      "Velocity Tracking Error (XY)")

fig1.savefig(OUTPUT_DIR / "poster_headline.png", dpi=180, bbox_inches="tight")
print("Saved poster_headline.png")

# ── Figure 2: Energy & Efficiency ────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("RB vs SQUAD — Energy & Mechanical Efficiency", fontsize=14, fontweight="bold")
fig2.subplots_adjust(wspace=0.35)

dual(axes2[0], "rew_joint_torques", "RB", "SQUAD", "Reward (less neg = better)", "Joint Torque Penalty")
dual(axes2[1], "rew_joint_acc",     "RB", "SQUAD", "Reward (less neg = better)", "Joint Acceleration Penalty")
dual(axes2[2], "rew_foot_slip",     "RB", "SQUAD", "Reward (less neg = better)", "Foot Slip Penalty")

fig2.savefig(OUTPUT_DIR / "poster_efficiency.png", dpi=180, bbox_inches="tight")
print("Saved poster_efficiency.png")

# ── Figure 3: Gait & locomotion quality ──────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle("RB vs SQUAD — Gait & Locomotion Quality", fontsize=14, fontweight="bold")
fig3.subplots_adjust(wspace=0.35)

dual(axes3[0], "rew_gait",          "RB", "SQUAD", "Reward",  "Gait Regularity")
dual(axes3[1], "rew_air_time",      "RB", "SQUAD", "Reward",  "Air Time (stepping quality)")
dual(axes3[2], "rew_foot_clearance","RB", "SQUAD", "Reward",  "Foot Clearance")

fig3.savefig(OUTPUT_DIR / "poster_gait.png", dpi=180, bbox_inches="tight")
print("Saved poster_gait.png")

# ── Figure 4: Convergence & stability ────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle("RB vs SQUAD — Convergence & Stability", fontsize=14, fontweight="bold")
fig4.subplots_adjust(wspace=0.35)

dual(axes4[0], "value_function_loss", "RB", "SQUAD", "Loss",  "Value Function Loss")
dual(axes4[1], "action_noise_std",    "RB", "SQUAD", "Std",   "Action Noise Std (policy confidence)")
dual(axes4[2], "term_body_contact",   "RB", "SQUAD", "Rate",  "Body Contact Terminations (falls)")

fig4.savefig(OUTPUT_DIR / "poster_convergence.png", dpi=180, bbox_inches="tight")
print("Saved poster_convergence.png")

# ── Figure 5: Final-value bar chart summary ───────────────────────────────────
# Mean reward excluded — it's already the headline figure and dominates the scale.
# Value labels are printed on top of (or below, for negatives) every bar.
N = 100
metrics_bar = {
    "Linear Vel\nReward":    "rew_base_linear_vel",
    "Terrain\nLevel":        "terrain_level",
    "Vel Error\nXY":         "error_vel_xy",
    "Joint Torque\nPenalty": "rew_joint_torques",
    "Joint Acc\nPenalty":    "rew_joint_acc",
    "Foot Slip\nPenalty":    "rew_foot_slip",
    "Gait\nReward":          "rew_gait",
    "Air Time\nReward":      "rew_air_time",
}

labels, rb_vals, sq_vals = [], [], []
for label, col in metrics_bar.items():
    if col in rb.columns and col in squad.columns:
        labels.append(label)
        rb_vals.append(rb[col].iloc[-N:].mean())
        sq_vals.append(squad[col].iloc[-N:].mean())

x  = np.arange(len(labels))
w  = 0.32
fig5, ax5 = plt.subplots(figsize=(14, 7))

bars_rb    = ax5.bar(x - w/2, rb_vals,  w, label="RB",    color=RB_COLOR,    alpha=0.88, zorder=3)
bars_squad = ax5.bar(x + w/2, sq_vals,  w, label="SQUAD", color=SQUAD_COLOR, alpha=0.88, zorder=3)

# Value labels on each bar — above for positive, below for negative
def label_bars(bars, vals):
    for bar, val in zip(bars, vals):
        x_pos  = bar.get_x() + bar.get_width() / 2
        offset = 0.015 * (ax5.get_ylim()[1] - ax5.get_ylim()[0])
        if val >= 0:
            y_pos  = bar.get_height() + offset
            va     = "bottom"
        else:
            y_pos  = bar.get_height() - offset
            va     = "top"
        ax5.text(x_pos, y_pos, f"{val:.3f}", ha="center", va=va,
                 fontsize=7.5, fontweight="bold", zorder=4)

# Draw bars first so ylim is set, then label
fig5.canvas.draw()          # forces ylim to be computed
label_bars(bars_rb,    rb_vals)
label_bars(bars_squad, sq_vals)

ax5.set_xticks(x)
ax5.set_xticklabels(labels, fontsize=9)
ax5.set_ylabel("Avg value (last 100 iters)", fontsize=9)
ax5.set_title("Final Converged Metric Comparison — RB vs SQUAD\n"
              "(Mean Reward shown separately in headline figure)",
              fontsize=12, fontweight="bold")
ax5.legend(fontsize=11)
ax5.grid(True, axis="y", alpha=0.25, zorder=0)
ax5.axhline(0, color="black", linewidth=0.8, zorder=2)

# Add a little headroom so top labels aren't clipped
ymin, ymax = ax5.get_ylim()
ax5.set_ylim(ymin * 1.15 if ymin < 0 else ymin, ymax * 1.18)

fig5.tight_layout()
fig5.savefig(OUTPUT_DIR / "poster_summary_bars.png", dpi=180, bbox_inches="tight")
print("Saved poster_summary_bars.png")

plt.close("all")
print("\nAll done! Check the outputs/ folder.")