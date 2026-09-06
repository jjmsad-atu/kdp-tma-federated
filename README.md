# KDP-TMA: Enhancing Federated Learning via Key-Data Provenance and Trap-Model Auditing

Experimental codebase for the paper *"Enhancing Federated Learning Aggregation via Key-Data
Provenance and Trap-Model Auditing"* by Jenan Jader Msad, Al-Furat Al-Awsat Technical University.

## Project Structure

```
kdp_tma_starter/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── configs/
│   └── default.yaml             # Global experimental configuration
├── data/
│   └── partition.py             # Dirichlet non-IID partitioning
├── models/
│   ├── simple_cnn.py            # CNN for Fashion-MNIST
│   └── resnet18.py              # ResNet-18 for CIFAR-10
├── defenses/                    # Aggregation strategies (Week 2)
│   ├── fedavg.py                # Baseline: no defence
│   ├── krum.py                  # Blanchard et al. 2017
│   ├── trimmed_mean.py          # Yin et al. 2018
│   ├── median.py                # Yin et al. 2018
│   ├── fltrust.py               # Cao et al. 2021
│   ├── foolsgold.py             # Fung et al. 2020
│   └── kdp_tma.py               # OUR METHOD
├── attacks/                     # Adversary implementations (Week 3)
│   ├── sign_flip.py             # Byzantine: sign flipping
│   ├── gaussian_noise.py        # Byzantine: gradient noise
│   └── backdoor.py              # Backdoor: pattern trigger
├── scripts/
│   ├── client.py                # Flower client
│   ├── server.py                # Flower server orchestrator
│   ├── main.py                  # Experiment entry point
│   └── analyze.py               # Post-hoc statistics
├── logs/                        # Training logs (JSON per run)
├── results/                     # Aggregated results (CSV + figures)
└── notebooks/
    └── week1_starter.ipynb      # Colab notebook for Week 1
```

## Quick Start (Week 1)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare Dirichlet partitions
python data/partition.py --dataset fashion-mnist --clients 100 --alpha 0.5

# 3. Run FedAvg baseline (no attack, no defence)
python scripts/main.py \
    --dataset fashion-mnist \
    --defense fedavg \
    --attack none \
    --rounds 200 \
    --seed 42
```

Expected result: ~85-88% test accuracy on Fashion-MNIST after 200 rounds.

## Weekly Milestones

- **Week 1**: FedAvg on Fashion-MNIST reaches ≥85%.
- **Week 2**: All 6 baseline defences work on clean data.
- **Week 3**: Attacks succeed against undefended FedAvg (accuracy drops or ASR rises).
- **Week 4**: KDP-TMA integrated and outperforms FedAvg under attack.
- **Week 5**: Full Fashion-MNIST results (216 runs).
- **Week 6**: Full CIFAR-10 results (216 runs).
- **Week 7**: Statistical analysis + first paper draft.

## Reproducibility

Every experiment saves:
- Configuration (as YAML)
- Random seeds (as seed_state.pt)
- Per-round metrics (as JSON)
- Final model checkpoint (as .pt)

To reproduce: `python scripts/main.py --config logs/{run_id}/config.yaml`

## Contact

jenan.jader@atu.edu.iq
