"""Paper figures and numbers for Sec. VII from the compiler-verified campaign.

Writes, into `figures/campaign_2026_11/paper/`:
  f1_gain_equal_blocks.pdf per-task, per-k F1 gain at equal blocks (heatmap)
  pareto_fronts_p4c.pdf   Fig. A -- per-k Pareto fronts, independent vs joint + alignment
  effect_by_budget.pdf    Fig. B -- the A gain split into joint mapping and alignment, per M
                          and per feature group (k <= 8 | k >= 9)
  numbers.txt             every number quoted in the text and the agreement table

Blocks and stages are p4c's (load_campaign substitutes them and drops
p4c-infeasible designs). Run from the repo root:
  PYTHONPATH=. python scripts/paper_figures_campaign_2026_11.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.reporting.campaign_data import load_campaign, load_verification
from src.reporting.claims import _t_interval, twin_ladder

RUN = 'results/campaign_2026_11'
OUT = 'figures/campaign_2026_11/paper'
IND, JOINT, ALIGNED = 'independent', 'joint-off', 'joint-off-al'
PIPE_BLOCKS = 288  # 24 blocks x 12 stages

# Style of the CNSM figures (pareto_fronts.pdf / other_metrics.pdf).
BLUE, ORANGE, GREEN = 'steelblue', 'darkorange', 'seagreen'
# Orange means "joint + alignment" in the fronts and the overlap figure, so the
# joint-mapping step of the effect bars gets its own hue.
PURPLE = '#6a51a3'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 16, 'axes.titlesize': 17,
    'axes.facecolor': '#f0f0f0', 'axes.edgecolor': 'black', 'axes.grid': True,
    'grid.color': 'white', 'grid.linewidth': 1.2, 'axes.axisbelow': True,
    'legend.fontsize': 15, 'pdf.fonttype': 42,
})


def front(d, acc='acc_avg'):
    """Non-dominated (min blocks, max accuracy) points, by increasing blocks."""
    d = d.sort_values(['blocks', acc], ascending=[True, False])
    best, pts = -np.inf, []
    for b, a in zip(d['blocks'], d[acc]):
        if a > best:
            pts.append((b, a))
            best = a
    return np.array(pts)


PARETO_KS = [17, 15, 13, 11, 9, 7, 5, 3, 1]


def fig_pareto(df, metric='f1_avg'):
    fig, axs = plt.subplots(3, 3, figsize=(16, 6.0), sharex=True, sharey=True)
    for ax, k in zip(axs.flat, PARETO_KS):
        for arm, color, label in ((IND, BLUE, 'Per-model'),
                                  (ALIGNED, ORANGE, 'Shared + alignment')):
            p = front(df[(df['k'] == k) & (df['arm_slug'] == arm)], acc=metric)
            ax.plot(p[:, 0], p[:, 1], '-o', color=color, ms=5, lw=1.5, label=label)
        ax.set_title(f'k = {k} features')
        ax.set_xlim(0, 75)
        ax.set_ylim(0.72, 0.90)
    axs[0, 0].legend(loc='lower right')
    for ax in axs[-1]:
        ax.set_xlabel('Memory (TCAM blocks)')
    for ax in axs[:, 0]:
        ax.set_ylabel('Average F1-score')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'pareto_fronts_p4c.pdf'))
    plt.close(fig)


def front_gaps(df, arm=ALIGNED, base=IND, metric='f1_avg'):
    """Per k, compare the two pooled fronts as attainment functions: the
    score gain at equal blocks and the block saving at equal score,
    over the range both fronts cover."""
    rows = []
    for k in sorted(df['k'].unique()):
        fa = front(df[(df['k'] == k) & (df['arm_slug'] == arm)], acc=metric)
        fb = front(df[(df['k'] == k) & (df['arm_slug'] == base)], acc=metric)
        acc_at = lambda f, b: f[f[:, 0] <= b, 1].max() if (f[:, 0] <= b).any() else np.nan
        blocks_for = lambda f, a: f[f[:, 1] >= a, 0].min() if (f[:, 1] >= a).any() else np.nan
        grid = np.arange(max(fa[0, 0], fb[0, 0]), max(fa[-1, 0], fb[-1, 0]) + 1)
        gain = np.array([acc_at(fa, b) - acc_at(fb, b) for b in grid])
        levels = np.union1d(fa[:, 1], fb[:, 1])
        levels = levels[levels <= min(fa[-1, 1], fb[-1, 1])]
        save = np.array([blocks_for(fb, a) - blocks_for(fa, a) for a in levels])
        rel = np.array([(blocks_for(fb, a) - blocks_for(fa, a)) / blocks_for(fb, a) for a in levels])
        rows.append({'k': k, 'max_acc_gain_pp': 100 * np.nanmax(gain),
                     'min_acc_gain_pp': 100 * np.nanmin(gain),
                     'max_block_saving': np.nanmax(save), 'min_block_saving': np.nanmin(save),
                     'max_rel_saving': np.nanmax(rel),
                     'best_acc': fa[-1, 1], 'best_acc_base': fb[-1, 1]})
    return pd.DataFrame(rows)


GAIN_GRID = np.arange(0, 61)  # TCAM blocks
GAIN_MIN_SPLITS = 20


def attain(blocks, score):
    """Best score among designs with blocks <= b, for every b in GAIN_GRID (NaN if none)."""
    out = np.full(len(GAIN_GRID), np.nan)
    order = np.argsort(blocks)
    b, s = blocks[order], np.maximum.accumulate(score[order])
    idx = np.searchsorted(b, GAIN_GRID, side='right') - 1
    out[idx >= 0] = s[idx[idx >= 0]]
    return out


def split_gains(df, ks, task, arm=ALIGNED, base=IND):
    """Rows = data splits: best F1 of `arm` within b blocks minus that of `base`,
    pooling the budgets and the k values in `ks` of the split."""
    sub = df[df['k'].isin(list(ks))]
    rows = []
    for _, part in sub.groupby('split'):
        a, i = part[part['arm_slug'] == arm], part[part['arm_slug'] == base]
        rows.append(attain(a['blocks'].to_numpy(), a[f'f1_{task}'].to_numpy())
                    - attain(i['blocks'].to_numpy(), i[f'f1_{task}'].to_numpy()))
    return np.array(rows)


def gain_summary(g):
    """Mean over splits and its 95% t interval, where >= GAIN_MIN_SPLITS splits are defined."""
    mean, lo, hi = (np.full(len(GAIN_GRID), np.nan) for _ in range(3))
    for j in range(len(GAIN_GRID)):
        v = g[:, j][~np.isnan(g[:, j])]
        if len(v) >= GAIN_MIN_SPLITS:
            mean[j], _, _, lo[j], hi[j] = _t_interval(v, 0.95)
    return 100 * mean, 100 * lo, 100 * hi


def _runs(blocks, sign):
    """'lo-hi(+)' runs of consecutive blocks with the same sign."""
    out = []
    for b, s in zip(blocks, sign):
        if out and b == out[-1][1] + 1 and s == out[-1][2]:
            out[-1][1] = b
        else:
            out.append([b, b, s])
    return ', '.join(f'{a}-{b}({s})' for a, b, s in out) or 'none'


def fig_gain(df):
    """Per-task, per-k F1 gain of joint + alignment over independent at equal TCAM
    blocks, as a heatmap (rows k, columns blocks). Diverging blue (independent
    better) - grey - orange (joint better), the arm colours of the other figures.
    Dots: the split-level 95% CI excludes 0. Hatched: fewer than GAIN_MIN_SPLITS
    splits have a design of both mappings within that many blocks."""
    cmap = LinearSegmentedColormap.from_list(
        'ind_joint', ['#2b6a9e', '#9ec3e0', '#e8e8e8', '#f7b267', '#c45a00'])
    ks = np.arange(1, 18)
    edges_x = np.append(GAIN_GRID, GAIN_GRID[-1] + 1) - 0.5
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.6))
    rows = []
    for ax, task, title, lim in ((axs[0], 'app', 'a) Application identification', 20),
                                 (axs[1], 'ddos', 'b) DDoS detection', 2)):
        mean = np.full((len(ks), len(GAIN_GRID)), np.nan)
        sig = np.zeros_like(mean, bool)
        for i, k in enumerate(ks):
            m, lo, hi = gain_summary(split_gains(df, [k], task))
            mean[i] = m
            sig[i] = ~np.isnan(lo) & ((lo > 0) | (hi < 0))
            rows.append({'task': task, 'k': k, 'peak_pp': np.nanmax(m),
                         'peak_at': GAIN_GRID[np.nanargmax(m)], 'min_pp': np.nanmin(m),
                         'min_at': GAIN_GRID[np.nanargmin(m)],
                         'first_block': GAIN_GRID[~np.isnan(m)][0],
                         'gain_at_40': m[GAIN_GRID == 40][0],
                         'ci_excludes_0': _runs(GAIN_GRID[sig[i]], np.where(lo[sig[i]] > 0, '+', '-'))})
        ax.set_facecolor('white')
        ax.fill_between([edges_x[0], edges_x[-1]], 0.5, 17.5, facecolor='white',
                        edgecolor='#b0b0b0', hatch='///', lw=0, zorder=0)
        im = ax.pcolormesh(edges_x, np.arange(0.5, 18), np.ma.masked_invalid(mean), cmap=cmap,
                           norm=TwoSlopeNorm(0, -lim, lim), shading='flat', zorder=1)
        yy, xx = np.nonzero(sig)
        ax.plot(GAIN_GRID[xx], ks[yy], '.', color='black', ms=2.5, zorder=2)
        ax.grid(False)
        ax.set_xlim(edges_x[0], edges_x[-1])
        ax.set_ylim(0.5, 17.5)
        ax.set_yticks([1, 3, 5, 7, 9, 11, 13, 15, 17])
        ax.set_xlabel('Memory (TCAM blocks)')
        ax.set_ylabel('Number of features $k$')
        ax.set_title(title)
        clipped = np.nanmax(np.abs(mean)) > lim
        fig.colorbar(im, ax=ax, label='F1-score gain (pp)', extend='both' if clipped else 'neither')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'f1_gain_equal_blocks.pdf'))
    plt.close(fig)
    return pd.DataFrame(rows)


def ladder_with_total(df):
    """twin_ladder's two steps plus the total (aligned - independent) with the
    same split-level t interval, on the same inner-joined cells."""
    lad = twin_ladder(df)
    key = ['M', 'split', 'k']
    metrics = ['blocks', 'stage_depth', 'f1_app', 'f1_ddos']
    a = df[df['arm_slug'] == ALIGNED].set_index(key)[metrics]
    i = df[df['arm_slug'] == IND].set_index(key)[metrics]
    j = df[df['arm_slug'] == JOINT].set_index(key)[metrics]
    cells = a.index.intersection(i.index).intersection(j.index)
    d = (a.loc[cells] - i.loc[cells]).reset_index()
    for m in metrics:
        lo, hi, mean = [], [], []
        for M in lad['M']:
            part = d if M == 'all' else d[d['M'] == M]
            mu, _, _, l, h = _t_interval(part.groupby('split')[m].mean().to_numpy(), 0.95)
            mean.append(mu); lo.append(l); hi.append(h)
        lad[f'total_{m}'], lad[f'total_{m}_ci_low'], lad[f'total_{m}_ci_high'] = mean, lo, hi
    return lad


K_SPLIT = 8  # few features: k <= K_SPLIT; many: k > K_SPLIT


def fig_effects(df):
    """Per budget, two stacked bars (few / many features): the joint-mapping and
    alignment steps, and the total with its split-level 95% CI."""
    groups = [(f'$k \\leq {K_SPLIT}$', df[df['k'] <= K_SPLIT], None),
              (f'$k \\geq {K_SPLIT + 1}$', df[df['k'] > K_SPLIT], '///')]
    lads = [(name, ladder_with_total(part), hatch) for name, part, hatch in groups]
    labels = ['15', '25', '35', '50', '75', r'$\infty$']
    x = np.arange(len(labels))
    w, off = 0.36, 0.2
    panels = [('blocks', 'a) TCAM blocks', 'Change (blocks)', 1),
              ('stage_depth', 'b) Pipeline stages', 'Change (stages)', 1),
              ('f1_app', 'c) App. identification F1', 'Change (pp)', 100),
              ('f1_ddos', 'd) DDoS detection F1', 'Change (pp)', 100)]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (m, title, ylabel, s) in zip(axs, panels):
        for g, (_, lad, hatch) in enumerate(lads):
            lad = lad[lad['M'].astype(str) != 'all']
            xs = x + (2 * g - 1) * off
            share = lad[f'step_sharing_{m}'].to_numpy() * s
            align = lad[f'step_alignment_{m}'].to_numpy() * s
            # Diverging stack: positive parts build up from 0, negative parts down.
            pos, neg = np.zeros(len(x)), np.zeros(len(x))
            for vals, color in ((share, PURPLE), (align, GREEN)):
                bottom = np.where(vals >= 0, pos, neg)
                ax.bar(xs, vals, w, bottom=bottom, color=color, hatch=hatch,
                       edgecolor='white', lw=0, zorder=2)
                pos += np.where(vals >= 0, vals, 0)
                neg += np.where(vals < 0, vals, 0)
            tot = lad[f'total_{m}'].to_numpy() * s
            lo = lad[f'total_{m}_ci_low'].to_numpy() * s
            hi = lad[f'total_{m}_ci_high'].to_numpy() * s
            ax.errorbar(xs, tot, yerr=[tot - lo, hi - tot], fmt='D', color='black', ms=4,
                        capsize=2, lw=1.0, zorder=3)
        ax.axhline(0, color='black', lw=0.8, zorder=2)
        ax.axvline(1.5, color='grey', lw=0.8, ls='--', zorder=1)  # tight | ample budgets
        ax.set_xticks(x, labels)
        ax.set_xlabel('Memory budget M (blocks)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    handles = [Patch(color=PURPLE, label='Sharing'),
               Patch(color=GREEN, label='Threshold alignment'),
               Patch(facecolor='grey', label=lads[0][0]),
               Patch(facecolor='grey', hatch='///', edgecolor='white', label=lads[1][0]),
               plt.Line2D([], [], marker='D', color='black', ls='', label='Total (95% CI)')]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, 'effect_by_budget.pdf'))
    plt.close(fig)


def shared_fraction(df):
    ind = df[df['arm_slug'] == IND].copy()
    ind['shared'] = [len(set(a.split(';')) & set(b.split(';'))) / k
                     for a, b, k in zip(ind['features_app'], ind['features_ddos'], ind['k'])]
    return ind.groupby('k')['shared'].describe(percentiles=[0.25, 0.5, 0.75])


def agreement(v):
    for c in ('model_stage_depth', 'p4c_stage_depth', 'model_blocks', 'p4c_blocks'):
        v[c] = pd.to_numeric(v[c], errors='coerce')
    v = v[v['arm_slug'].isin([IND, JOINT, ALIGNED])]
    rows = {}
    for arm, part in list(v.groupby('arm_slug')) + [('all', v)]:
        rows[arm] = {
            'designs': len(part),
            'stage_depth_exact': (part['model_stage_depth'] == part['p4c_stage_depth']).mean(),
            'blocks_exact': (part['model_blocks'] == part['p4c_blocks']).mean(),
            **{f'verdict_{k}': int(n) for k, n in part['verdict'].value_counts().items()},
        }
    misses = v[v['verdict'].isin(['UNDER', 'OVER'])]
    return pd.DataFrame(rows).T, misses


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_campaign(RUN)
    df = df[df['arm_slug'].isin([IND, JOINT, ALIGNED])].copy()
    df['acc_avg'] = (df['acc_app'] + df['acc_ddos']) / 2
    df['f1_avg'] = (df['f1_app'] + df['f1_ddos']) / 2

    fig_pareto(df)
    gain = fig_gain(df)
    lad = ladder_with_total(df)
    fig_effects(df)
    overlap = shared_fraction(df)
    agree, misses = agreement(load_verification(RUN))

    binding = pd.read_csv('figures/campaign_2026_11/10_budget_binding.csv')
    align = pd.read_csv('figures/campaign_2026_11/11_alignment_twins_alignment_stats.csv')
    counts = pd.read_csv('figures/campaign_2026_11/11_alignment_twins_counts.csv')
    unl = df[np.isinf(df['M'])]

    with open(os.path.join(OUT, 'numbers.txt'), 'w', encoding='utf-8') as f:
        out = lambda *a: print(*a, file=f)
        pd.set_option('display.width', 200)
        out('== Effect ladder (step_sharing = joint - independent, step_alignment = aligned - joint,'
            ' total = aligned - independent; f1 in fractions)')
        cols = ['M', 'n_cells'] + [c for c in lad.columns
                                   if c.startswith(('step_', 'total_', 'mean_')) and 'ci' not in c]
        out(lad[cols].round(3).T.to_string())
        gaps = front_gaps(df)
        out('\n== Pareto front gaps on average F1 (Fig. pareto), joint + alignment vs independent'
            ' (positive = joint better)')
        out(gaps.round(3).to_string())
        for name, part in (('plotted k', gaps[gaps['k'].isin(PARETO_KS)]),
                           ('plotted k >= 7', gaps[gaps['k'].isin(PARETO_KS) & (gaps['k'] >= 7)]),
                           ('all k >= 7', gaps[gaps['k'] >= 7]), ('all k >= 4', gaps[gaps['k'] >= 4])):
            out(f'  {name}: max gain pp range {part["max_acc_gain_pp"].min():.2f}..'
                f'{part["max_acc_gain_pp"].max():.2f} (median {part["max_acc_gain_pp"].median():.2f}); '
                f'max rel saving {part["max_rel_saving"].min():.3f}..{part["max_rel_saving"].max():.3f} '
                f'(median {part["max_rel_saving"].median():.3f}); max block saving '
                f'{part["max_block_saving"].min():.0f}..{part["max_block_saving"].max():.0f}')
        out('\n== Pareto front gaps on average accuracy (previous version of the figure)')
        out(front_gaps(df, metric='acc_avg').round(3).to_string())
        out('\n== F1 gain at equal blocks (Fig. gain): mean over splits of best F1 within b blocks,'
            ' joint + alignment minus independent, pp')
        out(gain.round(2).to_string())
        out('\n== Mean test accuracy / F1 by arm (all designs)')
        out(df.groupby('arm_slug')[['acc_app', 'f1_app', 'acc_ddos', 'f1_ddos']].mean().round(4).to_string())
        out('\n== Compiler agreement')
        out(agree.to_string())
        out('miss stage error (p4c - model):', (misses['p4c_stage_depth'] - misses['model_stage_depth'])
            .value_counts().sort_index().to_dict())
        out('miss block error (p4c - model):', (misses['p4c_blocks'] - misses['model_blocks'])
            .describe().round(2).to_dict())
        out('\n== Shared features / k, independent arm')
        out(overlap.round(2).to_string())
        out('\n== Budget binding share')
        out(binding.to_string())
        out('\n== Alignment statistics')
        out(align.round(3).to_string())
        out(counts.to_string())
        out('\n== Unlimited budget (M = inf), p4c blocks and stages')
        out(unl.groupby('arm_slug')['blocks'].describe(percentiles=[.25, .5, .75, .95]).round(1).to_string())
        out(unl.groupby('arm_slug')['stage_depth'].describe().round(2).to_string())
        out('share at the 12-stage limit:', (unl['stage_depth'] == 12).groupby(unl['arm_slug']).mean()
            .round(3).to_dict())
        out(f'median blocks as share of the pipe ({PIPE_BLOCKS}):',
            (unl.groupby('arm_slug')['blocks'].median() / PIPE_BLOCKS).round(3).to_dict())
    print('wrote', OUT)


if __name__ == '__main__':
    sys.exit(main())
