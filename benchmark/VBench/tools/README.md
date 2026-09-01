# Manga2Animate evaluation tools

The scripts are kept in one small, searchable tool directory and use verb-led
names. They fall into four workflow groups:

- Preparation: prepare_six_case_run.py, prepare_kick_run.py, and
  prepare_baseline_run.py validate inputs and create symlink-only workspaces.
- Metrics: modified_dynamic_degree.py, both DINOv3 evaluators, and
  evaluate_i2v_altcorr.py run supplemental metrics.
- Runners: run_lovehina_metric.py, run_baseline_metric.py, and the two
  run-when-idle shell scripts isolate GPU jobs and record audits.
- Aggregation: the aggregate, combine, export, generate, and summarize scripts
  create case-equal summaries and reports.

Run scripts from the parent VBench/ directory so the local package and
relative runtime layout resolve consistently. Generated data belongs in
evaluation_runs/, which is intentionally ignored.
