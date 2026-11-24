# t=23 — Multi-Channel Kinematics & Burst-Rhythm Anomaly Verifier

Zero-dependency Python 3.12 submission for the 0.09 SOL zero-gas bounty (issue `incidere/solana-agent-bounties#8`).

Run: `bash run.sh` — fully offline, deterministic, no installs.

Design: channel-invariant features (rates normalized by each session's own median tick interval and median burst gap); fixed z-score acceptance bounds calibrated on the calibration split only; 2 interaction regimes × 4 classes; exact FAR/FRR and N×N swap matrix.

See `RESULT.txt` for verified holdout output.
