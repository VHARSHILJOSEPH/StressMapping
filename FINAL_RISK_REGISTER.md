# FINAL RISK REGISTER

| Risk | Severity | Evidence | Action |
|------|----------|----------|--------|
| Stress could influence baseline if calibration occurs during stress (early windows) | MEDIUM | Audit notes that `ProtectedDynamicBaseline` still accepts windows during `BASELINE` phase without explicit stress check; however UI filters by SQI and stress label. | Ensure UI disables acceptance of stress windows during calibration (already enforced). |
| Sampling‑rate mismatch between training (WESAD) and hardware (25 Hz) may affect feature fidelity | LOW | Documented limitation; no test failures observed. | Consider resampling or retraining with hardware‑collected data for production. |
| Lack of persistent personal baseline storage → loss on restart | LOW | Baseline held only in memory; reset on app restart clears it. | Add optional persistence (JSON) for long‑term studies. |
| No concrete VR integration tests | INFORMATIONAL | No Phase 8 artifacts found. | Plan and execute VR‑stress experiments before publication. |
| Missing detailed hardware‑setup guide (pin wiring, sensor placement) | INFORMATIONAL | No such documentation in repo. | Add hardware assembly guide to improve reproducibility. |
