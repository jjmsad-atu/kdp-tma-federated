"""Statistical re-analysis requested by the reviewer (comments #6 and #9).
Uses only the released per-run CSVs. Outputs stats_revision.csv/.json and a printed report.

Primary set : 21 cells = 24 minus the three cells whose target is BoT-IoT at its natural attack rate (99.6%).
Sensitivity : all 24 cells; leave-one-dataset-out (drop every cell that involves dataset d).
Uncertainty : hierarchical crossed bootstrap - resample source datasets and target datasets independently
              (pigeonhole bootstrap for a crossed design, Owen 2007), then resample seeds within each kept cell.
Tests       : one-sample Wilcoxon signed-rank on per-cell recovery (H0: median 0), Holm correction within each family,
              matched-pairs rank-biserial r as effect size.
Seed level  : within-cell seed SD vs between-cell SD of recovery; share of cells whose 3 seeds agree in sign.
"""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon

D = 'data/'
CELL = ['src', 'tgt', 'prior']
DS = ['ton', 'unsw', 'bot', 'cic']
B = 4000


def load():
    p1 = pd.read_csv(D + 'sfda_ids_phase1__real_metrics.csv')
    p1 = p1[p1.featset == 'no_artifacts']
    p2 = pd.read_csv(D + 'sfda_ids_phase2__real_metrics.csv')
    p3 = pd.read_csv(D + 'sfda_ids_phase3__real_metrics.csv')
    for p in (p1, p2, p3):
        p['prior'] = p.prior.astype(str)
    fam = {}
    # family A: SFDA methods (Phase 1)
    fam['A_sfda'] = (p1, ['bn_adapt', 'tent@0.001', 'shot@0.0001', 'aad', 'em_prior'])
    # family B: anchor methods, clean anchor (Phase 3, contam 0) + NP threshold (Phase 2, anchor 1000, clean)
    a = p3[p3.contam == 0.0]
    np2 = p2[(p2.method == 'np_threshold') & (p2.anchor == 1000) & (p2.contam == 0.0)].copy()
    ref2 = p2[p2.method.isin(['source_only', 'oracle'])]
    fam['B_anchor_clean'] = (pd.concat([a, np2]), ['maha_pre_plain', 'maha_plain', 'autoencoder', 'iforest',
                                                   'bara_fused', 'align_plain_head', 'np_threshold'])
    # family C: anchor methods at 5% contamination
    c = p3[p3.contam == 0.05]
    ref3 = p3[p3.method.isin(['source_only', 'oracle']) & (p3.contam == 0.0)]
    fam['C_anchor_5pct'] = (pd.concat([c[~c.method.isin(['source_only', 'oracle'])], ref3]),
                           ['maha_pre_plain', 'maha_plain', 'autoencoder', 'iforest', 'bara_fused'])
    return fam


def seed_table(df, methods):
    """seed x cell x method F1 array, with source_only and oracle."""
    df = df[df.method.isin(methods + ['source_only', 'oracle'])]
    W = df.pivot_table(index=CELL + ['seed'], columns='method', values='macro_f1', aggfunc='mean')
    return W


def recovery_from(W, methods):
    G = W.groupby(level=CELL).mean()
    gap = G.oracle - G.source_only
    return G[methods].sub(G.source_only, axis=0).div(gap, axis=0)


def primary_mask(idx):
    return ~((idx.get_level_values('tgt') == 'bot') & (idx.get_level_values('prior') == 'natural'))


def crossed_boot(W, methods, keep, rng):
    """Hierarchical crossed bootstrap of the median recovery."""
    cells = W.groupby(level=CELL).size().index
    cells = cells[keep(cells)]
    seeds_of = {c: W.loc[c].index.values for c in cells}
    arr = {c: W.loc[c] for c in cells}
    out = np.full((B, len(methods)), np.nan)
    for b in range(B):
        S = rng.choice(DS, 4, replace=True); T = rng.choice(DS, 4, replace=True)
        recs = []
        for s in S:
            for t in T:
                if s == t:
                    continue
                for c in cells:
                    if c[0] == s and c[1] == t:
                        sd = rng.choice(seeds_of[c], len(seeds_of[c]), replace=True)
                        m = arr[c].loc[sd].mean()
                        g = m['oracle'] - m['source_only']
                        recs.append(((m[methods] - m['source_only']) / g).values)
        if recs:
            out[b] = np.median(np.vstack(recs), axis=0)
    return out


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def rank_biserial(x):
    x = np.asarray(x); x = x[x != 0]
    r = pd.Series(np.abs(x)).rank().values
    return float((r[x > 0].sum() - r[x < 0].sum()) / r.sum()) if len(x) else np.nan


def main():
    rng = np.random.default_rng(2026)
    rows = []
    for fname, (df, methods) in load().items():
        W = seed_table(df, methods)
        R = recovery_from(W, methods)
        prim = R[primary_mask(R.index)]
        bt = crossed_boot(W, methods, primary_mask, rng)
        # naive (cell-level iid) bootstrap for comparison
        nb = np.array([np.median(prim.values[rng.integers(0, len(prim), len(prim))], axis=0) for _ in range(B)])
        pv = [wilcoxon(prim[m].dropna()).pvalue for m in methods]
        ph = holm(pv)
        # seed-level: per-seed recovery
        Gs = W.copy()
        rec_seed = Gs[methods].sub(Gs.source_only, axis=0).div((Gs.oracle - Gs.source_only), axis=0)
        within_sd = rec_seed.groupby(level=CELL).std().loc[prim.index]
        diff_sign = np.sign(Gs[methods].sub(Gs.source_only, axis=0)).groupby(level=CELL).agg(lambda s: abs(s.sum()) == len(s)).loc[prim.index]
        for j, m in enumerate(methods):
            lodo = {}
            for d in DS:
                k = (prim.index.get_level_values('src') != d) & (prim.index.get_level_values('tgt') != d)
                lodo[d] = float(prim[m][k].median())
            rows.append(dict(family=fname, method=m, n_cells=int(prim[m].notna().sum()),
                             median_primary=float(prim[m].median()),
                             crossed_ci_lo=float(np.nanpercentile(bt[:, j], 2.5)), crossed_ci_hi=float(np.nanpercentile(bt[:, j], 97.5)),
                             naive_ci_lo=float(np.percentile(nb[:, j], 2.5)), naive_ci_hi=float(np.percentile(nb[:, j], 97.5)),
                             median_all24=float(R[m].median()),
                             wilcoxon_p=float(pv[j]), holm_p=float(ph[j]), rank_biserial=rank_biserial(prim[m].dropna()),
                             lodo_min=min(lodo.values()), lodo_max=max(lodo.values()), **{f'lodo_drop_{d}': v for d, v in lodo.items()},
                             seed_sd_within_median=float(within_sd[m].median()), between_cell_sd=float(prim[m].std()),
                             seed_sign_agree=float(diff_sign[m].mean()),
                             cells_harmed=int((prim[m] < -0.05 * 1).sum())))
    T = pd.DataFrame(rows)
    T.to_csv('stats_revision.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['family', 'method', 'n_cells', 'median_primary', 'crossed_ci_lo', 'crossed_ci_hi', 'naive_ci_lo', 'naive_ci_hi',
            'median_all24', 'holm_p', 'rank_biserial', 'lodo_min', 'lodo_max', 'seed_sd_within_median', 'between_cell_sd', 'seed_sign_agree']
    print(T[cols].round(3).to_string(index=False))
    print(T[['method'] + [c for c in T if c.startswith('lodo_drop')]].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
