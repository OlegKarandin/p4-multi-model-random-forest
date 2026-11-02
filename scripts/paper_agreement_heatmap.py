"""Paper figure: joint error (stage depth x TCAM blocks, predicted - p4c) of the
resource model and of the public-specification baseline, over the campaign
designs that p4c placed. Replaces Table II's exact / within-1 rows.

The baseline is scripts/paper_resource_baseline.py's; the model side is the
verification record's model_stage_depth / model_blocks.

Usage:
  python scripts/paper_agreement_heatmap.py [--run results/campaign_2026_11]
Writes figures/campaign_2026_11/paper/agreement_heatmap.pdf and prints the
both-exact share of each panel.
"""
import argparse
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_resource_baseline import load_run  # noqa: E402

OUT = 'figures/campaign_2026_11/paper'
# Error bins; the outer ones collect everything beyond them.
BLOCK_ERRORS, STAGE_ERRORS = np.arange(-6, 4), np.arange(-2, 2)
# The blue end of fig_gain's diverging scale (paper_figures_campaign_2026_11.py),
# extended to a darker navy so that the 0-0 cell stands out.
CMAP = LinearSegmentedColormap.from_list('blue', ['#e8e8e8', '#9ec3e0', '#2b6a9e', '#0b2a4a'])
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 16, 'axes.titlesize': 17,
    'axes.edgecolor': 'black', 'pdf.fonttype': 42,
})


def error_shares(rows, est):
    """Share (%) of designs per (stage error, block error) cell, clipped to the bins."""
    clip = lambda v, bins: int(np.clip(v, bins[0], bins[-1]))
    counts = Counter((clip(r[est + '_s'] - r['p4c_s'], STAGE_ERRORS),
                      clip(r[est + '_b'] - r['p4c_b'], BLOCK_ERRORS)) for r in rows)
    shares = np.array([[100 * counts[(s, b)] / len(rows) for b in BLOCK_ERRORS]
                       for s in STAGE_ERRORS])
    assert abs(shares.sum() - 100) < 1e-9
    return shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='results/campaign_2026_11')
    args = ap.parse_args()
    # p4c_b is None for designs p4c could not place: no block error exists.
    rows = [r for r in load_run(args.run) if r['p4c_b'] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2), sharey=True)
    for ax, est, title in zip(axes, ('model', 'base'), ('a) Resource model', 'b) Baseline')):
        shares = error_shares(rows, est)
        im = ax.pcolormesh(np.append(BLOCK_ERRORS, BLOCK_ERRORS[-1] + 1) - .5,
                           np.append(STAGE_ERRORS, STAGE_ERRORS[-1] + 1) - .5,
                           np.ma.masked_where(shares == 0, shares), cmap=CMAP,
                           norm=LogNorm(0.01, 100), edgecolors='white', linewidth=1.0)
        exact = shares[list(STAGE_ERRORS).index(0), list(BLOCK_ERRORS).index(0)]
        ax.text(0, 0, f'{exact:.0f}', ha='center', va='center', fontsize=11, color='white')
        print(f'{est}: both exact {exact:.1f}% of {len(rows)} designs')
        ax.set_facecolor('white')
        ax.set_title(title)
        ax.set_xlabel('Block error')
        ax.set_xticks([-6, -4, -2, 0, 3])
        ax.set_xticklabels(['≤ −6', '−4', '−2', '0', '≥ 3'])
        ax.set_yticks(STAGE_ERRORS)
        ax.set_yticklabels(['≤ −2', '−1', '0', '1'])
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Stage error')
    cb = fig.colorbar(im, ax=axes, label='Designs (%)', pad=0.02)
    cb.set_ticks([0.01, 0.1, 1, 10, 100])
    cb.set_ticklabels(['0.01', '0.1', '1', '10', '100'])
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'agreement_heatmap.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
