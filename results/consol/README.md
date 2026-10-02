# Consolidated-benchmark results

Copies of the result files from `data/consol/` (which is gitignored because it
also holds staged video frames). No videos are included.

- `manifest.json` — the 439 annotated clips with human labels
- `stage_signals.json` — Stage-1/2 signals per clip
- `vlmplaus_<model>__f{4,8}.json` — VLM plausibility / completion / hidden-property scores
- `*__variant-plaus_debias.json` — the physics-error ("debiased") question
- `mcq_<model>__f8.json` — specialist forced-choice probabilities per physics family
- `pairwise_*.json` — pairwise judge runs
- `system_eval_{pairs,all}.json` — output of `backend/scripts/system_eval.py`

Scripts: `backend/scripts/system_eval.py`, `paper_tables.py`, `consol_analyses.py`.
