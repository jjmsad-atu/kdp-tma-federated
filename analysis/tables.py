import pandas as pd, numpy as np, json, sys
D = 'data/'
p1 = pd.read_csv(D + 'sfda_ids_phase1__real_metrics.csv'); p1['prior'] = p1.prior.astype(str)
p2 = pd.read_csv(D + 'sfda_ids_phase2__real_metrics.csv'); p2['prior'] = p2.prior.astype(str)
p3 = pd.read_csv(D + 'sfda_ids_phase3__real_metrics.csv'); p3['prior'] = p3.prior.astype(str)
t1 = pd.read_csv(D + 'sfda_ids_phase1__real_per_type.csv'); t1['prior'] = t1.prior.astype(str)
out = {}
# ---- integrity
k1 = ['featset', 'seed', 'src', 'tgt', 'prior', 'method']; k2 = ['seed', 'src', 'tgt', 'prior', 'method', 'anchor', 'contam']; k3 = ['seed', 'src', 'tgt', 'prior', 'method', 'contam']
print('dups', p1.duplicated(k1).sum(), p2.duplicated(k2).sum(), p3.duplicated(k3).sum())
print('p1 cells', p1.groupby(['featset', 'prior']).size().to_dict())
print('NaN macro_f1', p1.macro_f1.isna().sum(), p2.macro_f1.isna().sum(), p3.macro_f1.isna().sum())
rng = np.random.RandomState(0)
def boot_med(v, B=5000):
    v = np.asarray(v.dropna()); 
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    m = np.median(rng.choice(v, (B, len(v)), replace=True), axis=1)
    return (float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))
cell = ['src', 'tgt', 'prior']
# ---- Phase 1 (RQ1): per featset, per cell mean over seeds
W = p1.pivot_table(index=['featset', 'seed'] + cell, columns='method', values='macro_f1')
W['tent_tuned'] = W[[c for c in W if c.startswith('tent@')]].max(1)
W['shot_tuned'] = W[[c for c in W if c.startswith('shot@')]].max(1)
G = W.groupby(['featset'] + cell).mean()
G['gap'] = G.oracle - G.source_only
meth1 = ['bn_adapt', 'tent@0.001', 'shot@0.0001', 'aad', 'em_prior', 'tent_tuned', 'shot_tuned']
rows = []
for fs in ['all', 'no_artifacts']:
    g = G.loc[fs]
    for m in meth1:
        rec = (g[m] - g.source_only) / g.gap
        md, lo, hi = boot_med(rec)
        rows.append(dict(featset=fs, method=m, n_cells=int(rec.notna().sum()), median_recovery=md, ci_lo=lo, ci_hi=hi,
                         max_recovery=float(rec.max()), cells_ge_half=int((rec >= 0.5).sum()),
                         cells_harmed=int((g[m] < g.source_only - 0.05).sum()), cells_helped=int((g[m] > g.source_only + 0.05).sum()),
                         median_f1=float(g[m].median())))
T2 = pd.DataFrame(rows); out['T2_phase1_sfda'] = T2
ref = G[['source_only', 'oracle', 'gap']].groupby('featset').median(); out['T2_ref'] = ref
print('\nT2 Phase1 SFDA'); print(T2.round(3).to_string(index=False)); print(ref.round(3))
# source in-domain
print('in-domain F1 by src', p1[p1.method == 'source_only'].groupby(['featset', 'src']).src_in_domain_f1.mean().round(3).to_dict())
# false-alarm flood & collapse (per run)
C = p1[p1.method.isin(['source_only', 'bn_adapt', 'tent@0.001', 'shot@0.0001', 'aad', 'em_prior'])].copy()
C['flood'] = C.fpr > 0.2; C['collapse'] = (C.pred_attack_rate < 0.01) & (C.true_attack_rate > 0.03)
T3 = C.groupby(['featset', 'method'])[['flood', 'collapse', 'fpr', 'attack_recall']].agg({'flood': 'mean', 'collapse': 'mean', 'fpr': 'median', 'attack_recall': 'median'})
out['T3_failure_modes'] = T3; print('\nT3'); print(T3.round(3))
# EM prior estimates
E = p1[p1.method == 'em_prior'].groupby(['featset'] + cell)[['est_prior', 'true_attack_rate']].mean()
E['abs_err'] = (E.est_prior - E.true_attack_rate).abs()
out['T4_em'] = E; print('\nEM abs error median by featset', E.groupby('featset').abs_err.median().round(3).to_dict(), 'max', E.abs_err.max().round(3))
print(E.loc['no_artifacts'].round(3).to_string())
# artifact ablation source-only
A = G.source_only.unstack('featset').groupby(['src', 'tgt']).mean(); print('\nartifact ablation src-only mean over priors'); print(A.round(3)); out['ablation_artifacts'] = A
# unseen types (no_artifacts, natural+0.05)
U = t1[(t1.featset == 'no_artifacts') & t1.method.isin(['source_only', 'tent@0.001', 'shot@0.0001', 'aad', 'bn_adapt'])].groupby(['method', 'seen_in_source']).recall.mean().unstack()
out['unseen'] = U; print('\nunseen types'); print(U.round(3))
# ---- Phase 2 (RQ2): AUROC and anchor baselines
p2['variant'] = np.where(p2.anchor > 0, p2.method + '@' + p2.anchor.astype(str) + '/c' + p2.contam.astype(str), p2.method)
G2 = p2.groupby(cell + ['variant'])[['macro_f1', 'fpr', 'attack_recall', 'auroc']].mean()
F2 = G2.macro_f1.unstack(); gap2 = F2.oracle - F2.source_only
R2 = F2.sub(F2.source_only, axis=0).div(gap2, axis=0)
rows = []
for v in [c for c in F2 if c not in ('oracle', 'source_only')]:
    md, lo, hi = boot_med(R2[v])
    rows.append(dict(variant=v, median_recovery=md, ci_lo=lo, ci_hi=hi, cells_ge_0_7=int((R2[v] >= 0.7).sum()),
                     median_fpr=float(G2.fpr.unstack()[v].median()), median_recall=float(G2.attack_recall.unstack()[v].median()),
                     median_auroc=float(G2.auroc.unstack()[v].median())))
T5 = pd.DataFrame(rows); out['T5_phase2'] = T5; print('\nT5 Phase2'); print(T5.round(3).to_string(index=False))
au = G2.auroc.unstack()[['source_only', 'oracle']]
print('src AUROC median', au.source_only.median().round(3), 'cells <0.5:', int((au.source_only < 0.5).sum()), '/', len(au), 'min', au.source_only.min().round(3))
out['auroc_src'] = au
# ---- Phase 3 (RQ3/RQ4)
p3m = p3[p3.method != '_anchor_kept_frac']
ref3 = p3m[p3m.method.isin(['source_only', 'oracle'])].groupby(cell + ['method']).macro_f1.mean().unstack()
M3 = p3m[~p3m.method.isin(['source_only', 'oracle'])].groupby(cell + ['contam', 'method'])[['macro_f1', 'fpr', 'attack_recall', 'auroc']].mean().reset_index()
M3 = M3.merge(ref3.reset_index(), on=cell); M3['rec'] = (M3.macro_f1 - M3.source_only) / (M3.oracle - M3.source_only)
rows = []
for (c, m), g in M3.groupby(['contam', 'method']):
    md, lo, hi = boot_med(g.rec)
    rows.append(dict(contam=c, method=m, median_recovery=md, ci_lo=lo, ci_hi=hi, cells_ge_0_7=int((g.rec >= 0.7).sum()),
                     median_f1=float(g.macro_f1.median()), median_fpr=float(g.fpr.median()), median_recall=float(g.attack_recall.median()), median_auroc=float(g.auroc.median())))
T6 = pd.DataFrame(rows); out['T6_phase3'] = T6
print('\nT6 Phase3'); print(T6.round(3).to_string(index=False))
kept = p3[p3.method == '_anchor_kept_frac'].groupby('contam').macro_f1.agg(['mean', 'min']); print('kept', kept.round(3).to_dict())
# BoT natural prior check
print('\nBoT natural cells, phase3 c=0 median F1 by method:'); print(M3[(M3.tgt == 'bot') & (M3.prior == 'natural') & (M3.contam == 0)].groupby('method').macro_f1.median().round(3).to_dict())
# Excluding BoT-natural sensitivity for the headline anchor result
ex = M3[~((M3.tgt == 'bot') & (M3.prior == 'natural'))]
print('\nexcluding bot-natural, maha_pre_plain median rec by contam:', ex[ex.method == 'maha_pre_plain'].groupby('contam').rec.median().round(3).to_dict())
pd.to_pickle(out, 'tables.pkl')
