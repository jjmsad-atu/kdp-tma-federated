# sfda-nids-evaluation

Code and per-run results for "Why Source-Free Adaptation Fails for Cross-Network Intrusion Detection".

Notebooks run on Google Colab (GPU). Datasets: NetFlow-v2 from Kaggle (dhoogla/nftoniotv2, dhoogla/nfunswnb15v2, dhoogla/nfbotiotv2, dhoogla/nfcsecicids2018v2). Raw data are not redistributed.

## Contents
- `*.ipynb` (root): executed notebooks with outputs (feasibility, Phases 1-4).
- `frozen/SFDA_IDS_phase4_revision.ipynb`: the Phase 4 notebook exactly as frozen before execution (no outputs).
  SHA-256: `ba5cb463c4a9d7e7399342cf1e461ba9eec61d93f1d84b9651e2064f4ecb0177`
  Verify with `sha256sum frozen/SFDA_IDS_phase4_revision.ipynb`.
  The executed copy in the root contains outputs, so its hash differs; its code cells are identical.
- `results/phase1..phase4/`: per-run result files (one row per seed, direction, prior and method).
- `analysis/`: scripts that produce the tables, figures and the 21-cell robustness analysis (`stats_revision.py`).

## Reproduce the paper's tables
```
mkdir -p data fig && cp results/phase1/* results/phase2/* results/phase3/* data/
python analysis/tables.py          # Tables 2-4
python analysis/figs.py            # Figures 1-3
python analysis/stats_revision.py  # Table 5
```
Phase 4 tables (6-8) are printed by the last cell of the Phase 4 notebook.
