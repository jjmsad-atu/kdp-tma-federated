import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8a85', 'axes.labelcolor': '#2b2b29', 'xtick.color': '#55554f', 'ytick.color': '#55554f',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 300})
INK, MUTED, BLUE = '#2b2b29', '#8a8a85', '#2a78d6'
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']; MK = ['o', 's', '^', 'D', 'v']
D = 'data/'; cell = ['src', 'tgt', 'prior']
rng = np.random.RandomState(1)
def strip(ax, groups, labels, ref_lines, ylab):
    for i, v in enumerate(groups):
        v = np.asarray(v); x = i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x, v, s=16, color=BLUE, alpha=0.75, linewidths=0, zorder=3)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=4)
    for y, txt in ref_lines:
        ax.axhline(y, color=MUTED, linewidth=1, linestyle='--', zorder=1)
        ax.text(len(groups) - 0.5, y, ' ' + txt, color=MUTED, fontsize=7.5, va='center', ha='left', clip_on=False)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels); ax.set_xlim(-0.5, len(groups) - 0.5); ax.set_ylabel(ylab); ax.grid(axis='x', visible=False)

# Fig 1 — Phase 1 recovery per cell (no_artifacts)
p1 = pd.read_csv(D + 'sfda_ids_phase1__real_metrics.csv'); p1['prior'] = p1.prior.astype(str)
W = p1[p1.featset == 'no_artifacts'].pivot_table(index=['seed'] + cell, columns='method', values='macro_f1')
W['shot_tuned'] = W[[c for c in W if c.startswith('shot@')]].max(axis=1); W['tent_tuned'] = W[[c for c in W if c.startswith('tent@')]].max(axis=1)
G = W.groupby(cell).mean(); gap = G.oracle - G.source_only
ms = ['bn_adapt', 'tent@0.001', 'shot@0.0001', 'aad', 'em_prior', 'tent_tuned', 'shot_tuned']
lab = ['BN-Adapt', 'TENT', 'SHOT', 'AaD', 'EM prior', 'TENT\n(tuned*)', 'SHOT\n(tuned*)']
fig, ax = plt.subplots(figsize=(6.8, 3.2))
strip(ax, [((G[m] - G.source_only) / gap).values for m in ms], lab, [(0, 'no adaptation'), (0.5, 'half of the gap')], 'Fraction of gap recovered')
ax.set_ylim(-0.75, 1.0)
fig.text(0.01, 0.01, '* learning rate chosen with target labels (unreachable upper bound). Dots: 24 cells (12 directions × 2 priors), bar: median.', fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 0.9, 1)); fig.savefig('fig/fig1_sfda_recovery.png'); fig.savefig('fig/fig1_sfda_recovery.pdf'); plt.close(fig)

# Fig 2 — AUROC diagnosis (Phase 3, clean anchor)
p3 = pd.read_csv(D + 'sfda_ids_phase3__real_metrics.csv'); p3['prior'] = p3.prior.astype(str)
A = p3[(p3.contam == 0) | p3.method.isin(['source_only', 'oracle'])].groupby(cell + ['method']).auroc.mean().unstack()
ms2 = ['source_only', 'bara_align_head', 'maha_plain', 'maha_pre_plain', 'oracle']
lab2 = ['Source head', 'Source head after\nbenign alignment', 'Mahalanobis\n(post-ReLU)', 'Mahalanobis\n(pre-ReLU)', 'Oracle\n(target labels)']
fig, ax = plt.subplots(figsize=(6.8, 3.2))
strip(ax, [A[m].values for m in ms2], lab2, [(0.5, 'chance')], 'Target AUROC'); ax.set_ylim(0, 1.03)
fig.text(0.01, 0.01, 'Dots: 24 cells; bar: median. Anchor methods use 1,000 clean benign target flows.', fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 0.93, 1)); fig.savefig('fig/fig2_auroc_diagnosis.png'); fig.savefig('fig/fig2_auroc_diagnosis.pdf'); plt.close(fig)

# Fig 3 — contamination curve (median recovery + 95% bootstrap CI)
ref = p3[p3.method.isin(['source_only', 'oracle'])].groupby(cell + ['method']).macro_f1.mean().unstack()
M = p3[~p3.method.isin(['source_only', 'oracle', '_anchor_kept_frac'])].groupby(cell + ['contam', 'method']).macro_f1.mean().reset_index().merge(ref.reset_index(), on=cell)
M['rec'] = (M.macro_f1 - M.source_only) / (M.oracle - M.source_only)
ms3 = ['maha_pre_plain', 'maha_plain', 'autoencoder', 'iforest', 'bara_fused']
lab3 = ['Mahalanobis (pre-ReLU)', 'Mahalanobis (post-ReLU)', 'Autoencoder', 'Isolation Forest', 'BARA (fused)']
cs = [0.0, 0.01, 0.05, 0.10]; brng = np.random.RandomState(0)
fig, ax = plt.subplots(figsize=(6.8, 3.6))
for k, (m, l) in enumerate(zip(ms3, lab3)):
    med, lo, hi = [], [], []
    for c in cs:
        v = M[(M.method == m) & (M.contam == c)].rec.values
        b = np.median(brng.choice(v, (5000, len(v))), axis=1); med.append(np.median(v)); lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
    x = np.arange(len(cs)) + (k - 2) * 0.06
    ax.errorbar(x, med, yerr=[np.array(med) - lo, np.array(hi) - med], color=COL[k], marker=MK[k], markersize=6, linewidth=2, capsize=0, elinewidth=1, label=l)
ax.set_xticks(range(len(cs))); ax.set_xticklabels(['0%', '1%', '5%', '10%']); ax.set_xlim(-0.3, 3.3)
ax.set_xlabel('Attack contamination of the benign anchor (1,000 flows)'); ax.set_ylabel('Median fraction of gap recovered')
ax.axhline(0, color=MUTED, linewidth=1, linestyle='--'); ax.legend(frameon=False, fontsize=7.5, loc='upper right')
ax.grid(axis='x', visible=False)
fig.text(0.01, 0.01, 'Whiskers: 95% bootstrap CI of the median over 24 cells (12 directions × 2 priors), 3 seeds each.', fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig('fig/fig3_contamination.png'); fig.savefig('fig/fig3_contamination.pdf'); plt.close(fig)
print('ok')
