"""Experiment infrastructure for the SWIFT paper.

Library modules:
    runner_base: Shared dataclasses, model training, utilities
    runner_controlled: Controlled experiment runner (S1-S9)
    runner_gradual: Gradual drift experiment runner (S10)
    data_loader: Dataset downloading, loading, and preprocessing
    drift: Drift injection framework (scenarios S1-S10)
    baselines: Baseline methods (PSI/CSI, SSI, KS, Raw W1, MMD, BBSD, Decker)
    evaluation: Evaluation metrics (TPR@FPR, AUROC, Spearman rho)
    ablations: Ablation variants (A1-A5)

Runner scripts (executable via ``python -m experiments.<name>`` or directly):
    run_taiwan_credit: Controlled drift on Taiwan Credit (UCI id=350)
    run_bank_marketing: Controlled drift on Bank Marketing (UCI id=222)
    run_home_credit: Controlled drift on Home Credit (Kaggle)
    run_lending_club: Temporal drift on Lending Club (Kaggle)
    run_calibration: Type I error calibration study
    run_power_analysis: Detection rate vs sample size
    run_multi_seed: Multi-seed stability with confidence intervals
    run_gradual_drift: S10 gradual drift detection delay
    run_ablations: Ablation experiments A1-A5
    run_all.sh: Bash orchestrator that runs all experiments sequentially
"""

